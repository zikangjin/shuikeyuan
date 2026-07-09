from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import math
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.errors import WindowError
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, box
from shapely.ops import nearest_points, substring

from .data_loader import (
    BUILDING_ID_CANDIDATES,
    RasterItem,
    first_existing_field,
    load_buildings_for_village,
    load_river_network,
    load_roads,
    load_section_table_trim_ranges,
    load_sections,
    load_village_boundary,
    scan_flood_rasters,
)
from .export_utils import compact_list, safe_filename, write_outputs
from .raster_utils import (
    crop_raster_to_temp,
    open_raster_crs,
    raster_bounds,
    sample_points,
    section_value_stats,
    zonal_max_mean,
)
from .section_analysis import choose_analysis_crs, nearest_section


@dataclass
class AnalysisResult:
    summary: dict
    building_details: pd.DataFrame
    flooded_details: pd.DataFrame
    section_distances: pd.DataFrame
    map_payload: dict
    xlsx_path: Path
    csv_path: Path


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "是"}


def _bbox_overlap_score(reference_bounds, target_bounds) -> float:
    ref_minx, ref_miny, ref_maxx, ref_maxy = reference_bounds
    tgt_minx, tgt_miny, tgt_maxx, tgt_maxy = target_bounds
    ref_area = max((ref_maxx - ref_minx) * (ref_maxy - ref_miny), 0.0)
    ix = max(0.0, min(ref_maxx, tgt_maxx) - max(ref_minx, tgt_minx))
    iy = max(0.0, min(ref_maxy, tgt_maxy) - max(ref_miny, tgt_miny))
    intersection = ix * iy
    score = intersection / ref_area if ref_area > 0 else 0.0
    cx = (ref_minx + ref_maxx) / 2
    cy = (ref_miny + ref_maxy) / 2
    if tgt_minx <= cx <= tgt_maxx and tgt_miny <= cy <= tgt_maxy:
        score = max(score, 0.25)
    return score


def _candidate_crs_values(config: dict, buildings: gpd.GeoDataFrame) -> list:
    values = [
        config.get("target_projected_crs"),
        config.get("section_crs"),
        config.get("building_crs"),
        config.get("auxiliary_crs"),
        buildings.crs,
        "EPSG:4548",
        "EPSG:4547",
        "EPSG:4549",
        "EPSG:4527",
        "EPSG:4528",
        "EPSG:3857",
        "EPSG:4326",
        "EPSG:4490",
    ]
    seen = set()
    candidates = []
    for value in values:
        if value in (None, ""):
            continue
        try:
            crs = CRS.from_user_input(value)
        except Exception:
            continue
        key = crs.to_string()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(crs)
    return candidates


def _raster_crs_from_config(first_raster: Path, config: dict, buildings: gpd.GeoDataFrame, logger):
    embedded = open_raster_crs(first_raster)
    if embedded is not None:
        logger.info(f"栅格文件自带 CRS：{embedded}")
        return embedded

    configured = str(config.get("raster_crs") or "").strip()
    if configured:
        raster_crs = CRS.from_user_input(configured)
        logger.info(f"使用用户配置的栅格 CRS：{raster_crs}")
        return raster_crs

    bounds = raster_bounds(first_raster)
    candidates = _candidate_crs_values(config, buildings)
    matches = []
    for candidate in candidates:
        try:
            projected = buildings.to_crs(candidate)
            score = _bbox_overlap_score(projected.total_bounds, bounds)
        except Exception:
            continue
        if score > 0:
            matches.append((score, candidate))

    if matches:
        matches.sort(key=lambda item: item[0], reverse=True)
        score, raster_crs = matches[0]
        logger.warning(
            f"栅格缺少 CRS，已根据村庄建筑物范围与栅格范围自动推断为 {raster_crs}，匹配分数 {score:.3f}。"
        )
        return raster_crs

    candidate_text = "、".join(crs.to_string() for crs in candidates) or "无"
    raise ValueError(
        "栅格缺少 CRS，且自动推断失败。"
        f"已尝试候选坐标系：{candidate_text}。"
        "请至少为栅格或断面填写一个正确 CRS，或使用带 .prj/内嵌 CRS 的栅格。"
    )


def _select_previous_item(items: list[RasterItem], first_item: RasterItem, interval_hours: float):
    if first_item.parsed_time:
        target = first_item.parsed_time - timedelta(hours=interval_hours)
        before = [item for item in items if item.parsed_time and item.parsed_time <= first_item.parsed_time]
        if not before:
            return None
        return min(before, key=lambda item: abs((item.parsed_time - target).total_seconds()))
    if first_item.hour_value is not None:
        target_hour = first_item.hour_value - interval_hours
        candidates = [item for item in items if item.hour_value is not None and item.hour_value < first_item.hour_value]
        if not candidates:
            return None
        return min(candidates, key=lambda item: abs(item.hour_value - target_hour))
    target_order = first_item.order - max(1, round(interval_hours))
    for item in items:
        if item.order == target_order:
            return item
    return None


def _building_value_stats(buildings_in_raster_crs: gpd.GeoDataFrame, raster_path: Path, config: dict):
    geometries = list(buildings_in_raster_crs.geometry)
    fill_zero = _as_bool(config.get("treat_zero_as_nodata"), True)
    all_touched = _as_bool(config.get("all_touched"), False)
    geom_types = {str(t).lower() for t in buildings_in_raster_crs.geometry.geom_type.unique()}
    if all("point" in t for t in geom_types):
        return sample_points(geometries, raster_path, fill_none_with_zero=fill_zero)
    return zonal_max_mean(geometries, raster_path, all_touched=all_touched, fill_none_with_zero=fill_zero)


def _raster_max_value(raster_path: Path) -> float | None:
    with rasterio.open(raster_path) as src:
        data = src.read(1, masked=True)
    if data.size == 0:
        return None
    if np.ma.is_masked(data) and data.mask.all():
        return None
    value = float(data.max())
    return value if np.isfinite(value) else None


def _threshold_area_stats(raster_path: Path, geometry, threshold: float) -> dict:
    if geometry is None or geometry.is_empty:
        return {}
    try:
        with rasterio.open(raster_path) as src:
            window = from_bounds(*geometry.bounds, transform=src.transform)
            window = window.round_offsets().round_lengths()
            window = window.intersection(Window(0, 0, src.width, src.height))
            if window.width <= 0 or window.height <= 0:
                return {}
            data = src.read(1, window=window)
            transform = src.window_transform(window)
            inside = geometry_mask([geometry.__geo_interface__], out_shape=data.shape, transform=transform, invert=True)
            values = data[inside]
            if values.size == 0:
                return {}
            cell_area = abs(src.transform.a * src.transform.e)
    except (WindowError, ValueError):
        return {}

    ge_threshold = values >= threshold
    gt_zero = values > 0
    return {
        "inside_cells": int(values.size),
        "nonzero_cells": int(gt_zero.sum()),
        "threshold_cells": int(ge_threshold.sum()),
        "threshold_area_m2": float(ge_threshold.sum() * cell_area),
        "threshold_area_ratio_pct": float(ge_threshold.mean() * 100),
        "nonzero_area_ratio_pct": float(gt_zero.mean() * 100),
    }


def _prepare_raster_for_stats(item: RasterItem, buildings_in_raster_crs: gpd.GeoDataFrame, raster_crs, config: dict) -> Path:
    if not _as_bool(config.get("use_temp_cropped_tif"), True):
        return item.path
    output_dir = Path(config.get("temp_dir") or Path(config["output_dir"]) / "temp" / "rasters")
    scenario = str(config.get("scenario_name") or item.path.parent.name or "scenario")
    target = output_dir / scenario
    bounds = tuple(buildings_in_raster_crs.total_bounds)
    return crop_raster_to_temp(
        item.path,
        bounds,
        target,
        raster_crs=raster_crs,
        padding_pixels=int(config.get("raster_crop_padding_cells") or 2),
    )


def _value_column_name(value_type: str) -> str:
    return "水深" if "位" not in value_type else "水位"


def _merge_bounds(bounds_list: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    valid = [bounds for bounds in bounds_list if bounds]
    if not valid:
        return None
    minx = min(bounds[0] for bounds in valid)
    miny = min(bounds[1] for bounds in valid)
    maxx = max(bounds[2] for bounds in valid)
    maxy = max(bounds[3] for bounds in valid)
    dx = max(maxx - minx, 1.0)
    dy = max(maxy - miny, 1.0)
    pad = max(dx, dy) * 0.06
    return minx - pad, miny - pad, maxx + pad, maxy + pad


def _map_bounds(buildings: gpd.GeoDataFrame, sections: gpd.GeoDataFrame, frames: list[dict]) -> tuple[float, float, float, float] | None:
    bounds_list: list[tuple[float, float, float, float]] = []
    if buildings is not None and not buildings.empty:
        bounds_list.append(tuple(buildings.total_bounds))
    if sections is not None and not sections.empty:
        bounds_list.append(tuple(sections.total_bounds))
    for frame in frames:
        path = frame.get("raster_path")
        if not path:
            continue
        try:
            bounds_list.append(raster_bounds(Path(path)))
        except Exception:
            continue
    return _merge_bounds(bounds_list)


def _clip_lines_for_map(gdf: gpd.GeoDataFrame | None, bounds: tuple[float, float, float, float] | None) -> gpd.GeoDataFrame | None:
    if gdf is None or gdf.empty or bounds is None:
        return None
    minx, miny, maxx, maxy = bounds
    candidates = gdf.cx[minx:maxx, miny:maxy].copy()
    if candidates.empty:
        return None
    mask = candidates.intersects(box(minx, miny, maxx, maxy))
    clipped = candidates[mask].copy()
    return clipped if not clipped.empty else None


def _add_river_center_map_columns(sections: gpd.GeoDataFrame, analysis_crs, map_crs) -> gpd.GeoDataFrame:
    if sections is None or sections.empty:
        return sections
    if "section_river_center_analysis_x" not in sections.columns or "section_river_center_analysis_y" not in sections.columns:
        return sections
    points = []
    valid_indices = []
    for idx, row in sections.iterrows():
        x = row.get("section_river_center_analysis_x")
        y = row.get("section_river_center_analysis_y")
        if pd.isna(x) or pd.isna(y):
            continue
        try:
            points.append(Point(float(x), float(y)))
            valid_indices.append(idx)
        except (TypeError, ValueError):
            continue
    sections = sections.copy()
    sections["section_river_center_map_x"] = np.nan
    sections["section_river_center_map_y"] = np.nan
    if not points:
        return sections
    centers = gpd.GeoSeries(points, crs=analysis_crs).to_crs(map_crs)
    for idx, point in zip(valid_indices, centers):
        sections.at[idx, "section_river_center_map_x"] = point.x
        sections.at[idx, "section_river_center_map_y"] = point.y
    return sections


def _section_order_value(section_id: str) -> int:
    numbers = re.findall(r"\d+", str(section_id or ""))
    return int(numbers[-1]) if numbers else 0


def _measured_centerlines_from_sections(sections: gpd.GeoDataFrame, crs) -> gpd.GeoDataFrame | None:
    if sections is None or sections.empty:
        return None
    required = {"section_river_center_analysis_x", "section_river_center_analysis_y"}
    if not required.issubset(set(sections.columns)):
        return None
    groups: dict[str, list[tuple[int, float, float, str]]] = {}
    for _, row in sections.iterrows():
        x = row.get("section_river_center_analysis_x")
        y = row.get("section_river_center_analysis_y")
        if pd.isna(x) or pd.isna(y):
            continue
        try:
            x_float = float(x)
            y_float = float(y)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x_float) or not math.isfinite(y_float):
            continue
        section_id = str(row.get("section_id", ""))
        river_name = str(row.get("river_folder", "") or row.get("section_name", "") or "实测河道")
        groups.setdefault(river_name, []).append((_section_order_value(section_id), x_float, y_float, section_id))

    rows = []
    for river_name, points in groups.items():
        points.sort(key=lambda item: item[0])
        unique_points: list[tuple[float, float]] = []
        section_ids = []
        for _order, x, y, section_id in points:
            xy = (x, y)
            if not unique_points or xy != unique_points[-1]:
                unique_points.append(xy)
            section_ids.append(section_id)
        if len(unique_points) < 2:
            continue
        rows.append(
            {
                "river_name": river_name,
                "source": "section_table_river_centers",
                "section_ids": compact_list(section_ids, limit=50),
                "point_count": len(unique_points),
                "geometry": LineString(unique_points),
            }
        )
    if not rows:
        return None
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def _export_corrected_rivers(centerlines: gpd.GeoDataFrame | None, output_dir: Path, village_name: str, scenario_name: str, logger) -> Path | None:
    if centerlines is None or centerlines.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"校正后河道_{safe_filename(village_name)}_{safe_filename(scenario_name)}.geojson"
    try:
        centerlines.to_file(path, driver="GeoJSON", encoding="utf-8")
        logger.info(f"校正后河道已输出：{path}")
        return path
    except Exception as exc:
        logger.warning(f"校正后河道输出失败：{exc}")
        return None


def _union_geometry(gdf: gpd.GeoDataFrame | None):
    if gdf is None or gdf.empty:
        return None
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union


def _line_parts(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom] if geom.length > 0 else []
    if isinstance(geom, MultiLineString):
        return [part for part in geom.geoms if part.length > 0]
    if isinstance(geom, GeometryCollection):
        parts: list[LineString] = []
        for part in geom.geoms:
            parts.extend(_line_parts(part))
        return parts
    return []


def _point_parts(geom) -> list[Point]:
    if geom is None or geom.is_empty:
        return []
    geom_type = geom.geom_type
    if geom_type == "Point":
        return [geom]
    if geom_type == "MultiPoint":
        return list(geom.geoms)
    if geom_type in {"LineString", "LinearRing"}:
        coords = list(geom.coords)
        return [Point(coords[0]), Point(coords[-1])] if coords else []
    if isinstance(geom, GeometryCollection):
        points: list[Point] = []
        for part in geom.geoms:
            points.extend(_point_parts(part))
        return points
    return []


def _select_line_part(parts: list[LineString], reference_geom):
    valid = [part for part in parts if part.length > 0]
    if not valid:
        return None
    if reference_geom is None or reference_geom.is_empty:
        return max(valid, key=lambda part: part.length)
    return min(valid, key=lambda part: (part.distance(reference_geom), -part.length))


def _line_substring(line: LineString, start_m: float, end_m: float):
    start_m = max(0.0, min(float(start_m), line.length))
    end_m = max(0.0, min(float(end_m), line.length))
    if end_m - start_m <= 1e-6:
        return None
    geom = substring(line, start_m, end_m)
    parts = _line_parts(geom)
    return _select_line_part(parts, None)


def _measure_ranges_on_line(line: LineString, obstacle_geom) -> list[tuple[float, float]]:
    if obstacle_geom is None or obstacle_geom.is_empty or line is None or line.is_empty:
        return []
    try:
        intersection = line.intersection(obstacle_geom)
    except Exception:
        return []
    ranges: list[tuple[float, float]] = []
    for part in _line_parts(intersection):
        coords = list(part.coords)
        if not coords:
            continue
        start = float(line.project(Point(coords[0])))
        end = float(line.project(Point(coords[-1])))
        if abs(end - start) > 1e-6:
            ranges.append((min(start, end), max(start, end)))
    for point in _point_parts(intersection):
        measure = float(line.project(point))
        ranges.append((measure, measure))
    return sorted(ranges)


def _trim_line_before_buildings(
    line: LineString,
    building_geom,
    river_m: float | None,
    clearance_m: float,
):
    """Trim a section segment so it keeps the river point but stops before buildings."""

    if line is None or line.is_empty or river_m is None or building_geom is None or building_geom.is_empty:
        return line, 0.0, float(line.length) if line is not None and not line.is_empty else 0.0, False, ""
    river_m = max(0.0, min(float(river_m), line.length))
    clearance_m = max(0.0, float(clearance_m))
    ranges = _measure_ranges_on_line(line, building_geom)
    if not ranges:
        return line, 0.0, float(line.length), False, ""

    eps = 1e-6
    if any(start < river_m - eps and end > river_m + eps for start, end in ranges):
        return line, 0.0, float(line.length), False, "河中落在建筑范围内，未避让"

    start_m = 0.0
    end_m = float(line.length)
    before = [end for start, end in ranges if end <= river_m - eps]
    after = [start for start, end in ranges if start >= river_m + eps]
    if before:
        start_m = max(before) + clearance_m
    if after:
        end_m = min(after) - clearance_m

    if start_m >= river_m - eps:
        start_m = 0.0
    if end_m <= river_m + eps:
        end_m = float(line.length)
    if start_m <= eps and end_m >= line.length - eps:
        return line, 0.0, float(line.length), False, ""

    trimmed = _line_substring(line, start_m, end_m)
    if trimmed is None or trimmed.is_empty:
        return line, 0.0, float(line.length), False, "避让建筑后断面为空，未避让"
    return trimmed, start_m, end_m, True, f"避让建筑(clearance={clearance_m:g}m)"


def _table_river_center_point(table_range: dict | None, source_crs, target_crs):
    if not table_range:
        return None
    x = table_range.get("river_center_x")
    y = table_range.get("river_center_y")
    if x is None or y is None:
        return None
    try:
        point = Point(float(x), float(y))
    except (TypeError, ValueError):
        return None
    if source_crs is None or target_crs is None:
        return point
    try:
        source = CRS.from_user_input(source_crs)
        target = CRS.from_user_input(target_crs)
        if source != target:
            return gpd.GeoSeries([point], crs=source).to_crs(target).iloc[0]
    except Exception:
        return point
    return point


def _line_substring_from_table(line: LineString, table_range: dict, river_center_point=None, tolerance_m: float = 100.0):
    start_m = float(table_range["start_m"])
    end_m = float(table_range["end_m"])
    river_m = float(table_range.get("river_m", (start_m + end_m) / 2.0))
    if river_center_point is not None and not river_center_point.is_empty:
        distance_to_line = float(river_center_point.distance(line))
        if distance_to_line <= tolerance_m:
            center_on_line = float(line.project(river_center_point))
            return _line_substring(line, center_on_line - (river_m - start_m), center_on_line + (end_m - river_m)), True, distance_to_line
        return _line_substring(line, start_m, end_m), False, distance_to_line
    return _line_substring(line, start_m, end_m), False, None


def _river_measure_on_line(line: LineString, river_geom, reference_geom) -> float | None:
    if river_geom is None or river_geom.is_empty:
        return None
    intersection = line.intersection(river_geom)
    points = _point_parts(intersection)
    if points:
        if reference_geom is not None and not reference_geom.is_empty:
            point = min(points, key=lambda item: item.distance(reference_geom))
        else:
            point = points[len(points) // 2]
        return float(line.project(point))
    try:
        point_on_line, _point_on_river = nearest_points(line, river_geom)
        return float(line.project(point_on_line))
    except Exception:
        return None


def _road_measures_on_line(line: LineString, road_geom) -> list[float]:
    if road_geom is None or road_geom.is_empty:
        return []
    points = _point_parts(line.intersection(road_geom))
    measures = sorted({round(float(line.project(point)), 6) for point in points})
    return [measure for measure in measures if 0.0 <= measure <= line.length]


def _trim_line_by_road_river(line: LineString, road_geom, river_geom, reference_geom):
    river_m = _river_measure_on_line(line, river_geom, reference_geom)
    road_ms = _road_measures_on_line(line, road_geom)
    if river_m is None or not road_ms:
        return None
    eps = 1e-6
    before = [measure for measure in road_ms if measure < river_m - eps]
    after = [measure for measure in road_ms if measure > river_m + eps]
    if before and after:
        return _line_substring(line, max(before), min(after))
    if before:
        return _line_substring(line, max(before), line.length)
    if after:
        return _line_substring(line, 0.0, min(after))
    return None


def _section_trim_range_from_table(row, trim_ranges: dict[str, dict] | None) -> dict | None:
    if not trim_ranges:
        return None
    candidates: set[str] = set()
    qualifiers: set[str] = set()
    river_folder = str(row.get("river_folder", "") or "").strip()
    if river_folder:
        qualifiers.add(river_folder)
    for field in ("section_id", "section_name"):
        value = str(row.get(field, "") or "").strip()
        if not value:
            continue
        candidates.add(value)
        numbers = re.findall(r"\d+", value)
        if numbers:
            number = numbers[-1]
            stripped = number.lstrip("0") or "0"
            candidates.update({number, stripped, f"{int(stripped):02d}", f"横断{int(stripped):02d}", f"横断{stripped}"})
    source_path = str(row.get("source_path", "") or "").strip()
    if source_path:
        source = Path(source_path)
        stem = source.stem
        if source.parent.name.lower() == "dat":
            qualifiers.add(source.parent.parent.name)
        elif source.parent.name:
            qualifiers.add(source.parent.name)
        if stem:
            candidates.add(stem)
            if stem.isdigit():
                stripped = stem.lstrip("0") or "0"
                candidates.update({stripped, f"{int(stripped):02d}", f"横断{int(stripped):02d}", f"横断{stripped}"})
    for qualifier in qualifiers:
        for candidate in candidates:
            key = f"{qualifier}|{candidate}"
            if key in trim_ranges:
                return trim_ranges[key]
    for candidate in candidates:
        if candidate in trim_ranges:
            return trim_ranges[candidate]
    return None


def _trim_line_by_clip_geom(line: LineString, clip_geom, reference_geom):
    if clip_geom is None or clip_geom.is_empty:
        return None
    parts = _line_parts(line.intersection(clip_geom))
    return _select_line_part(parts, reference_geom)


def _trim_sections_for_context(
    sections: gpd.GeoDataFrame,
    config: dict,
    analysis_crs,
    reference_gdf: gpd.GeoDataFrame,
    village_boundary_gdf: gpd.GeoDataFrame | None,
    rivers_gdf: gpd.GeoDataFrame | None,
    roads_gdf: gpd.GeoDataFrame | None,
    table_trim_ranges: dict[str, dict] | None,
    logger,
    avoid_buildings_gdf: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    mode = str(config.get("section_trim_mode") or "road_river").strip().lower()
    if mode in {"", "none", "full", "完整断面", "不裁剪"}:
        result = sections.to_crs(analysis_crs).copy()
        result["original_length_m"] = result.geometry.length
        result["trimmed_length_m"] = result.geometry.length
        result["section_trim_method"] = "未裁剪"
        return result

    road_modes = {"auto", "自动", "road_river", "road-river", "路-河-路", "道路河道", "道路-河道"}
    context_modes = {"village_context", "building_context", "context", "村庄附近", "建筑附近"}
    buffer_m = float(config.get("section_trim_buffer_m") or 150)
    sections_projected = sections.to_crs(analysis_crs).copy()
    reference_geom = _union_geometry(reference_gdf.to_crs(analysis_crs))
    avoid_buildings_enabled = _as_bool(config.get("section_avoid_buildings"), True)
    building_clearance_m = float(config.get("section_building_clearance_m") or 2)
    building_avoid_source = avoid_buildings_gdf if avoid_buildings_gdf is not None else reference_gdf
    building_geom = (
        _union_geometry(building_avoid_source.to_crs(analysis_crs))
        if avoid_buildings_enabled and building_avoid_source is not None and not building_avoid_source.empty
        else None
    )
    village_geom = _union_geometry(village_boundary_gdf.to_crs(analysis_crs)) if village_boundary_gdf is not None else None
    river_geom = _union_geometry(rivers_gdf.to_crs(analysis_crs)) if rivers_gdf is not None else None
    road_geom = _union_geometry(roads_gdf.to_crs(analysis_crs)) if roads_gdf is not None else None
    fallback_geom = None
    fallback_name = ""
    if village_geom is not None and not village_geom.is_empty:
        fallback_geom = village_geom.buffer(buffer_m)
        fallback_name = f"村庄边界缓冲{buffer_m:g}m"
    elif reference_geom is not None and not reference_geom.is_empty:
        fallback_geom = reference_geom.buffer(buffer_m)
        fallback_name = f"参考建筑缓冲{buffer_m:g}m"

    trimmed_geoms = []
    methods = []
    original_lengths = []
    trimmed_lengths = []
    trim_start_values = []
    trim_end_values = []
    original_depth_values = []
    riverbed_elevation_values = []
    river_center_values = []
    boundary_elevation_values = []
    start_label_values = []
    end_label_values = []
    river_center_x_values = []
    river_center_y_values = []
    table_anchor_used_values = []
    table_anchor_distance_values = []
    river_center_analysis_x_values = []
    river_center_analysis_y_values = []
    road_ready = road_geom is not None and river_geom is not None
    table_ready = bool(table_trim_ranges)
    table_anchor_tolerance_m = float(config.get("section_table_anchor_tolerance_m") or 100)
    if mode in road_modes and not road_ready and not table_ready:
        missing = []
        if road_geom is None:
            missing.append("道路/路网数据 road_path 或断面汇总表 section_table_path")
        if river_geom is None:
            missing.append("河流水系 river_network_path 或断面汇总表中的河道/河中备注")
        raise ValueError(
            "断面水深现在按“道路-河中-道路/末端”的断面段计算，"
            f"但缺少：{'、'.join(missing)}。请在高级参数中选择道路/路网数据或断面汇总表，"
            "或把 section_trim_mode 改为 village_context 才会使用村庄附近的临时裁剪。"
        )

    for _, row in sections_projected.iterrows():
        geom = row.geometry
        original_lengths.append(float(geom.length) if geom is not None and not geom.is_empty else 0.0)
        best_part = None
        method = "原始断面"
        table_range = _section_trim_range_from_table(row, table_trim_ranges)
        row_start = None
        row_end = None
        row_original_depth = None
        row_riverbed_elevation = None
        row_river_center = None
        row_boundary_elevation = None
        row_start_label = ""
        row_end_label = ""
        row_river_center_x = None
        row_river_center_y = None
        row_table_anchor_used = False
        row_table_anchor_distance = None
        table_center_point = _table_river_center_point(table_range, sections.crs, analysis_crs)
        row_river_center_analysis_x = table_center_point.x if table_center_point is not None and not table_center_point.is_empty else None
        row_river_center_analysis_y = table_center_point.y if table_center_point is not None and not table_center_point.is_empty else None
        for line in _line_parts(geom):
            candidate = None
            candidate_method = ""
            if mode in road_modes and table_range is not None:
                candidate, anchor_used, anchor_distance = _line_substring_from_table(
                    line,
                    table_range,
                    table_center_point,
                    table_anchor_tolerance_m,
                )
                candidate_method = table_range.get("section_table_method", "汇总表道路-河中-道路/末端")
                if anchor_used:
                    candidate_method = f"{candidate_method}-河中坐标锚定"
                row_start = table_range["start_m"]
                row_end = table_range["end_m"]
                row_original_depth = table_range.get("section_original_depth_m")
                row_riverbed_elevation = table_range.get("section_riverbed_elevation_m")
                row_river_center = table_range.get("river_m")
                row_boundary_elevation = table_range.get("section_boundary_elevation_m")
                row_start_label = table_range.get("section_start_label", "")
                row_end_label = table_range.get("section_end_label", "")
                row_river_center_x = table_range.get("river_center_x")
                row_river_center_y = table_range.get("river_center_y")
                row_table_anchor_used = anchor_used
                row_table_anchor_distance = anchor_distance
            if candidate is None and mode in road_modes and road_ready:
                candidate = _trim_line_by_road_river(line, road_geom, river_geom, reference_geom)
                candidate_method = "道路-河道-道路/末端"
            if candidate is None and mode in context_modes:
                candidate = _trim_line_by_clip_geom(line, fallback_geom, reference_geom)
                candidate_method = fallback_name
            if candidate is None:
                continue

            candidate_len_before_avoid = float(candidate.length)
            river_m_on_candidate = None
            if table_range is not None and row_start is not None and row_river_center is not None:
                river_m_on_candidate = float(row_river_center) - float(table_range["start_m"])
                if table_center_point is not None and not table_center_point.is_empty and table_center_point.distance(candidate) <= table_anchor_tolerance_m:
                    river_m_on_candidate = float(candidate.project(table_center_point))
            elif river_geom is not None and not river_geom.is_empty:
                river_m_on_candidate = _river_measure_on_line(candidate, river_geom, reference_geom)

            candidate, avoid_start, avoid_end, avoided_buildings, avoid_note = _trim_line_before_buildings(
                candidate,
                building_geom,
                river_m_on_candidate,
                building_clearance_m,
            )
            if avoided_buildings:
                candidate_method = f"{candidate_method or method}-{avoid_note}"
                if table_range is not None and row_start is not None:
                    table_start = float(table_range["start_m"])
                    row_start = table_start + avoid_start
                    row_end = table_start + avoid_end
                    if avoid_start > 1e-6:
                        row_start_label = "建筑前缘"
                    if avoid_end < candidate_len_before_avoid - 1e-6:
                        row_end_label = "建筑前缘"
            elif avoid_note:
                candidate_method = f"{candidate_method or method}-{avoid_note}"

            if best_part is None:
                best_part = candidate
                method = candidate_method or method
            elif reference_geom is None or reference_geom.is_empty:
                if candidate.length > best_part.length:
                    best_part = candidate
                    method = candidate_method or method
            elif candidate.distance(reference_geom) < best_part.distance(reference_geom):
                best_part = candidate
                method = candidate_method or method
        if best_part is None:
            if mode in road_modes:
                best_part = GeometryCollection()
                method = "未裁剪：未找到汇总表路-河中范围或道路-河道交点"
            else:
                best_part = geom
                method = "原始断面"
        trimmed_geoms.append(best_part)
        trimmed_lengths.append(float(best_part.length) if best_part is not None and not best_part.is_empty else 0.0)
        methods.append(method)
        trim_start_values.append(row_start)
        trim_end_values.append(row_end)
        original_depth_values.append(row_original_depth)
        riverbed_elevation_values.append(row_riverbed_elevation)
        river_center_values.append(row_river_center)
        boundary_elevation_values.append(row_boundary_elevation)
        start_label_values.append(row_start_label)
        end_label_values.append(row_end_label)
        river_center_x_values.append(row_river_center_x)
        river_center_y_values.append(row_river_center_y)
        table_anchor_used_values.append(row_table_anchor_used)
        table_anchor_distance_values.append(row_table_anchor_distance)
        river_center_analysis_x_values.append(row_river_center_analysis_x)
        river_center_analysis_y_values.append(row_river_center_analysis_y)

    result = sections_projected.copy()
    result["geometry"] = trimmed_geoms
    result["original_length_m"] = original_lengths
    result["trimmed_length_m"] = trimmed_lengths
    result["section_trim_method"] = methods
    result["section_trim_start_m"] = trim_start_values
    result["section_trim_end_m"] = trim_end_values
    result["section_river_center_m"] = river_center_values
    result["section_trim_start_label"] = start_label_values
    result["section_trim_end_label"] = end_label_values
    result["section_river_center_x"] = river_center_x_values
    result["section_river_center_y"] = river_center_y_values
    result["section_table_anchor_used"] = table_anchor_used_values
    result["section_table_anchor_distance_m"] = table_anchor_distance_values
    result["section_river_center_analysis_x"] = river_center_analysis_x_values
    result["section_river_center_analysis_y"] = river_center_analysis_y_values
    result["section_original_depth_m"] = original_depth_values
    result["section_riverbed_elevation_m"] = riverbed_elevation_values
    result["section_boundary_elevation_m"] = boundary_elevation_values
    changed = sum(1 for old, new in zip(original_lengths, trimmed_lengths) if new > 0 and new < old - 0.1)
    logger.info(f"断面裁剪完成：{changed}/{len(result)} 条断面被缩短；裁剪模式：{mode or 'auto'}。")
    return result


def _empty_like(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.iloc[0:0].copy()


def _section_reference_mode(config: dict) -> str:
    raw = str(config.get("section_reference_mode") or "first_flooded_buildings").strip()
    mapping = {
        "首次受淹建筑物": "first_flooded_buildings",
        "最深受淹建筑物": "first_flooded_max_building",
        "自动：优先村庄边界": "village_boundary_or_buildings",
        "仅村庄边界": "village_boundary",
        "建筑物整体": "buildings_geometry",
        "flooded": "first_flooded_buildings",
        "first_flooded": "first_flooded_buildings",
        "flooded_buildings": "first_flooded_buildings",
        "max_flooded": "first_flooded_max_building",
        "max_flooded_building": "first_flooded_max_building",
        "auto": "village_boundary_or_buildings",
        "boundary": "village_boundary",
        "buildings": "buildings_geometry",
    }
    return mapping.get(raw, raw)


def _first_flooded_reference(
    buildings: gpd.GeoDataFrame,
    first_details: pd.DataFrame | None,
    value_column: str,
    mode: str,
    logger,
) -> tuple[gpd.GeoDataFrame | None, str, list[str]]:
    if first_details is None or first_details.empty or "is_flooded" not in first_details.columns:
        return None, "", []

    flooded = first_details[first_details["is_flooded"]].copy()
    if flooded.empty:
        return None, "", []

    if mode == "first_flooded_max_building":
        max_idx = flooded[value_column].fillna(-np.inf).idxmax()
        position = first_details.index.get_loc(max_idx)
        selected = flooded.loc[[max_idx]]
        reference = buildings.iloc[[position]].copy()
        ids = selected["building_id"].astype(str).tolist()
        logger.info(f"最近断面距离计算使用首次受淹时最大{value_column.replace('max_', '')}建筑物：{ids[0]}")
        return reference, "first_flooded_max_building", ids

    positions = [first_details.index.get_loc(idx) for idx in flooded.index]
    reference = buildings.iloc[positions].copy()
    ids = flooded["building_id"].astype(str).tolist()
    logger.info(f"最近断面距离计算使用首次受淹建筑物整体几何，共 {len(reference)} 栋。")
    return reference, "first_flooded_buildings", ids


def run_analysis(config: dict, logger) -> AnalysisResult:
    required = ["building_path", "flood_folder"]
    for key in required:
        if not str(config.get(key) or "").strip():
            raise ValueError(f"参数为空：{key}")
    if not str(config.get("section_path") or config.get("section_paths") or "").strip():
        raise ValueError("参数为空：section_path / section_paths")

    output_dir = Path(config.get("output_dir") or "outputs")
    config["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    value_type = str(config.get("value_type") or "水深")
    value_label = _value_column_name(value_type)
    village_name = str(config.get("village_name") or "全部建筑物").strip() or "全部建筑物"
    scenario_name = str(config.get("scenario_name") or "").strip() or Path(config["flood_folder"]).name
    threshold = float(config.get("threshold") or 0.1)
    interval_hours = float(config.get("time_interval_hours") or 1)

    logger.info("开始读取建筑物数据。")
    buildings, village_field, filter_method = load_buildings_for_village(config, logger)
    id_field = str(config.get("building_id_field") or "").strip() or first_existing_field(buildings.columns, BUILDING_ID_CANDIDATES)
    if id_field and id_field in buildings.columns:
        building_ids = buildings[id_field].astype(str).tolist()
    else:
        id_field = "building_id"
        buildings = buildings.copy()
        buildings[id_field] = [str(i + 1) for i in range(len(buildings))]
        building_ids = buildings[id_field].tolist()
    logger.info(f"建筑物编号字段：{id_field}")

    logger.info("开始扫描淹没栅格时间序列。")
    rasters = scan_flood_rasters(Path(config["flood_folder"]), value_type, logger)
    raster_crs = _raster_crs_from_config(rasters[0].path, config, buildings, logger)
    config["raster_crs"] = raster_crs.to_string()
    logger.info(f"栅格 CRS：{raster_crs}")

    buildings_raster = buildings.to_crs(raster_crs)
    analysis_crs = choose_analysis_crs(buildings, raster_crs, config.get("target_projected_crs"), logger)
    logger.info(f"距离计算 CRS：{analysis_crs}")
    threshold_area_gdf = buildings_raster
    threshold_area_reference = "selected_buildings"
    if config.get("auxiliary_path") and village_name and village_name != "全部建筑物":
        try:
            threshold_area_gdf = load_village_boundary(
                Path(config["auxiliary_path"]),
                village_name,
                field=config.get("auxiliary_village_field") or None,
                encoding=config.get("auxiliary_encoding") or "gbk",
                crs_if_missing=config.get("auxiliary_crs"),
            ).to_crs(raster_crs)
            threshold_area_reference = "village_boundary"
        except Exception as exc:
            logger.warning(f"村域达阈值面积比例无法使用村庄边界，改用参与分析建筑物范围。原因：{exc}")
    threshold_area_geom = threshold_area_gdf.geometry.union_all()

    first_item: RasterItem | None = None
    first_details: pd.DataFrame | None = None
    first_area_stats: dict = {}
    max_value_at_first = None
    animation_frames: list[dict] = []
    stop_after_first = _as_bool(config.get("stop_after_first_flood"), True)
    animate_all_hours = _as_bool(config.get("animate_all_hours"), True)
    calculate_area_ratio_all_hours = _as_bool(config.get("calculate_area_ratio_all_hours"), False)

    for item in rasters:
        logger.info(f"分析时刻 {item.label}：{item.path.name}")
        try:
            stats_raster = _prepare_raster_for_stats(item, buildings_raster, raster_crs, config)
            area_stats = (
                _threshold_area_stats(item.path, threshold_area_geom, threshold)
                if first_item is None or calculate_area_ratio_all_hours
                else {}
            )
            if first_item is not None and stop_after_first and animate_all_hours:
                animation_frames.append(
                    {
                        "label": item.label,
                        "raster_path": str(stats_raster),
                        "flooded_count": "-",
                        "max_value": _raster_max_value(stats_raster),
                        **area_stats,
                    }
                )
                continue
            stats = _building_value_stats(buildings_raster, stats_raster, config)
        except Exception as exc:
            raise RuntimeError(f"栅格分析失败：{item.path}；原因：{exc}") from exc

        details = pd.DataFrame(
            {
                "building_id": building_ids,
                f"max_{value_label}": [row["max"] for row in stats],
                f"mean_{value_label}": [row["mean"] for row in stats],
            }
        )
        if village_field and village_field in buildings.columns:
            details[village_field] = buildings[village_field].astype(str).tolist()
        details["is_flooded"] = details[f"max_{value_label}"].fillna(0) >= threshold

        flooded_count = int(details["is_flooded"].sum())
        max_value = details[f"max_{value_label}"].max()
        animation_frames.append(
            {
                "label": item.label,
                "raster_path": str(stats_raster),
                "flooded_count": flooded_count,
                "max_value": float(max_value) if pd.notna(max_value) else None,
                **area_stats,
            }
        )
        logger.info(f"时刻 {item.label}：受淹建筑物 {flooded_count} / {len(details)}；最大{value_label} {max_value:.4f}")
        if flooded_count > 0:
            first_item = item
            first_details = details
            first_area_stats = area_stats
            max_value_at_first = float(max_value)
            logger.info(f"找到首次受淹时刻：{item.label}")
            if stop_after_first and not animate_all_hours:
                break

    if first_details is None:
        building_details = pd.DataFrame({"building_id": building_ids})
        flooded_details = pd.DataFrame(columns=["building_id", f"max_{value_label}", f"mean_{value_label}", "is_flooded"])
    else:
        building_details = first_details
        flooded_details = first_details[first_details["is_flooded"]].copy()

    logger.info("开始计算最近河道断面。")
    sections, section_id_field, section_name_field = load_sections(config, logger)
    reference_gdf = buildings
    reference_type = "selected_buildings_geometry"
    reference_building_ids: list[str] = []
    reference_mode = _section_reference_mode(config)
    note_parts: list[str] = []
    value_column = f"max_{value_label}"

    if reference_mode in {"first_flooded_buildings", "first_flooded_max_building"}:
        flooded_reference, flooded_reference_type, reference_building_ids = _first_flooded_reference(
            buildings,
            first_details,
            value_column,
            reference_mode,
            logger,
        )
        if flooded_reference is not None:
            reference_gdf = flooded_reference
            reference_type = flooded_reference_type
        else:
            logger.warning("没有可用于距离计算的受淹建筑物，最近断面改用参与分析建筑物整体几何。")
            reference_type = "selected_buildings_geometry_no_flood_fallback"
            note_parts.append("无受淹建筑物，最近断面改按参与分析建筑物整体几何计算。")
    elif reference_mode in {"village_boundary_or_buildings", "village_boundary"} and config.get("auxiliary_path") and village_name and village_name != "全部建筑物":
        try:
            reference_gdf = load_village_boundary(
                Path(config["auxiliary_path"]),
                village_name,
                field=config.get("auxiliary_village_field") or None,
                encoding=config.get("auxiliary_encoding") or "gbk",
                crs_if_missing=config.get("auxiliary_crs"),
            )
            reference_type = "village_boundary"
            logger.info("最近断面距离计算使用村庄边界整体几何。")
        except Exception as exc:
            if reference_mode == "village_boundary":
                raise ValueError(f"最近断面参考对象选择了“仅村庄边界”，但村庄边界不可用：{exc}") from exc
            logger.warning(f"村庄边界无法用于最近断面计算，改用参与分析建筑物整体几何。原因：{exc}")
    elif reference_mode == "village_boundary":
        raise ValueError("最近断面参考对象选择了“仅村庄边界”，但未提供辅助村庄边界数据。")
    if reference_type.startswith("selected_buildings_geometry"):
        logger.info("最近断面距离计算使用参与分析建筑物整体几何。")

    rivers_source = None
    if _as_bool(config.get("show_river_network"), True) or str(config.get("section_trim_mode") or "road_river").strip().lower() not in {"none", "full", "不裁剪", "完整断面"}:
        try:
            rivers_source = load_river_network(config, logger)
        except Exception as exc:
            logger.warning(f"河流水系读取失败，断面裁剪和动画河流叠加将跳过河流。原因：{exc}")

    roads_source = None
    if str(config.get("road_path") or "").strip():
        try:
            roads_source = load_roads(config, logger)
        except Exception as exc:
            logger.warning(f"道路/路网读取失败，断面将不按道路交点裁剪。原因：{exc}")

    village_boundary_for_trim = threshold_area_gdf.to_crs(analysis_crs) if threshold_area_reference == "village_boundary" else None
    sections_for_distance = _trim_sections_for_context(
        sections,
        config,
        analysis_crs,
        reference_gdf,
        village_boundary_for_trim,
        rivers_source,
        roads_source,
        load_section_table_trim_ranges(config, logger),
        logger,
        avoid_buildings_gdf=buildings,
    )

    nearest = nearest_section(reference_gdf, sections_for_distance, analysis_crs, section_id_field, section_name_field, reference_type)
    logger.info(f"最近断面：{nearest['section_id']}；距离 {nearest['distance_m']:.2f} m")
    section_distances = nearest["section_distances"]
    max_section_distance_m = float(config.get("max_section_distance_m") or 3000)
    nearest_within_limit = max_section_distance_m <= 0 or nearest["distance_m"] <= max_section_distance_m
    if not nearest_within_limit:
        logger.warning(
            f"最近断面距离 {nearest['distance_m']:.2f} m，超过最大允许距离 {max_section_distance_m:.2f} m；"
            "动画不显示远处断面，结果不提取该断面值。"
        )
        note_parts.append(
            f"最近候选断面 {nearest['section_id']} 距离 {nearest['distance_m']:.2f} m，"
            f"超过最大允许距离 {max_section_distance_m:.2f} m，未作为有效最近断面。"
        )
    map_section_count = int(config.get("map_nearby_section_count") or 8)
    if nearest_within_limit:
        section_indices = section_distances.head(map_section_count)["section_index"].tolist()
        nearby_sections = sections_for_distance.loc[section_indices].copy().to_crs(raster_crs)
        nearby_sections = _add_river_center_map_columns(nearby_sections, analysis_crs, raster_crs)
        distance_map = section_distances.set_index("section_index")["distance_m"].to_dict()
        nearby_sections["distance_m"] = [distance_map.get(idx) for idx in nearby_sections.index]
    else:
        nearby_sections = _empty_like(sections_for_distance).to_crs(raster_crs)
        nearby_sections = _add_river_center_map_columns(nearby_sections, analysis_crs, raster_crs)
    bounds_for_map_layers = _map_bounds(buildings_raster, nearby_sections, animation_frames)
    corrected_rivers_source = _measured_centerlines_from_sections(sections_for_distance, analysis_crs)
    corrected_river_path = None
    if _as_bool(config.get("export_corrected_river_network"), True):
        corrected_river_path = _export_corrected_rivers(corrected_rivers_source, output_dir, village_name, scenario_name, logger)
    corrected_rivers_for_map = None
    if corrected_rivers_source is not None and not corrected_rivers_source.empty:
        corrected_rivers_for_map = _clip_lines_for_map(corrected_rivers_source.to_crs(raster_crs), bounds_for_map_layers)
    rivers_for_map = None
    if _as_bool(config.get("show_river_network"), True):
        try:
            if rivers_source is not None:
                rivers_for_map = _clip_lines_for_map(rivers_source.to_crs(raster_crs), bounds_for_map_layers)
                if rivers_for_map is not None:
                    logger.info(f"动画河流水系叠加数量：{len(rivers_for_map)} 条。")
                else:
                    logger.warning("河流水系与当前动画范围没有重叠，动画将不显示河流。")
        except Exception as exc:
            logger.warning(f"河流水系叠加失败，动画将不显示河流。原因：{exc}")

    map_payload = {
        "buildings": buildings_raster[[id_field, "geometry"]].copy() if id_field in buildings_raster.columns else buildings_raster[["geometry"]].copy(),
        "reference_buildings": reference_gdf.to_crs(raster_crs)[[id_field, "geometry"]].copy()
        if reference_type.startswith("first_flooded") and id_field in reference_gdf.columns
        else None,
        "section_reference_type": reference_type,
        "rivers": rivers_for_map,
        "corrected_rivers": corrected_rivers_for_map,
        "sections": nearby_sections,
        "frames": animation_frames,
        "threshold": threshold,
        "section_buffer_m": float(config.get("section_buffer_m") or 5),
        "section_trim_mode": str(config.get("section_trim_mode") or "road_river"),
        "value_label": value_label,
        "village_name": village_name,
        "scenario_name": scenario_name,
        "nearest_section_id": nearest["section_id"] if nearest_within_limit else "",
        "first_flood_time": first_item.label if first_item else "",
    }

    current_section_value = None
    previous_section_value = None
    previous_label = None
    section_method = "未提取"
    if first_item is None:
        note_parts.append("所有时刻均未达到阈值。")
    elif not nearest_within_limit:
        section_method = "未提取：最近断面超过最大允许距离"
    else:
        nearest_section_geom_raster = gpd.GeoSeries([nearest["geometry_analysis_crs"]], crs=analysis_crs).to_crs(raster_crs).iloc[0]
        buffer_m = float(config.get("section_buffer_m") or 5)
        trim_method = str(nearest["section"].get("section_trim_method", "") or "").strip()
        trim_prefix = "" if trim_method in {"", "原始断面", "未裁剪"} else f"{trim_method}裁剪后"
        section_method = "点值" if "point" in nearest_section_geom_raster.geom_type.lower() else f"{trim_prefix}线缓冲区最大值(buffer={buffer_m:g}m)"
        current_raster = _prepare_raster_for_stats(first_item, gpd.GeoDataFrame(geometry=[nearest_section_geom_raster], crs=raster_crs), raster_crs, config)
        current_stats = section_value_stats(nearest_section_geom_raster, current_raster, method=section_method, buffer_m=buffer_m)
        current_section_value = current_stats.get("max")

        previous_item = _select_previous_item(rasters, first_item, interval_hours)
        if previous_item is None:
            note_parts.append("无前序时刻数据。")
        else:
            previous_label = previous_item.label
            previous_raster = _prepare_raster_for_stats(previous_item, gpd.GeoDataFrame(geometry=[nearest_section_geom_raster], crs=raster_crs), raster_crs, config)
            previous_stats = section_value_stats(nearest_section_geom_raster, previous_raster, method=section_method, buffer_m=buffer_m)
            previous_section_value = previous_stats.get("max")

    is_flooded = first_item is not None
    summary = {
        "情景": scenario_name,
        "村庄": village_name,
        "阈值": threshold,
        "参与分析的建筑物数量": len(buildings),
        "参与分析的建筑物清单": compact_list(building_ids),
        "是否受淹": "是" if is_flooded else "否",
        "首次受淹时间": first_item.label if first_item else "",
        "首次受淹时被淹建筑物数量": len(flooded_details),
        "首次受淹时被淹建筑物清单": compact_list(flooded_details["building_id"].tolist()) if not flooded_details.empty else "",
        f"首次受淹时最大{value_label}": max_value_at_first,
        "首次受淹时达阈值面积比例": first_area_stats.get("threshold_area_ratio_pct"),
        "首次受淹时达阈值面积_m2": first_area_stats.get("threshold_area_m2"),
        "达阈值面积统计对象": threshold_area_reference,
        "最近断面编号": nearest["section_id"] if nearest_within_limit else "",
        "最近断面名称": nearest["section_name"] if nearest_within_limit else "",
        "最近断面距离": nearest["distance_m"],
        "最近候选断面编号": nearest["section_id"],
        "最近候选断面名称": nearest["section_name"],
        "最近断面是否在距离阈值内": "是" if nearest_within_limit else "否",
        "最近断面最大允许距离_m": max_section_distance_m,
        "最近断面距离参考对象": nearest["reference_type"],
        "最近断面参考建筑物": compact_list(reference_building_ids) if reference_building_ids else "",
        "最近断面原始长度_m": nearest["section"].get("original_length_m", ""),
        "最近断面裁剪后长度_m": nearest["section"].get("trimmed_length_m", ""),
        "最近断面裁剪方法": nearest["section"].get("section_trim_method", ""),
        "最近断面裁剪起点桩距_m": nearest["section"].get("section_trim_start_m", ""),
        "最近断面裁剪终点桩距_m": nearest["section"].get("section_trim_end_m", ""),
        "最近断面河中桩距_m": nearest["section"].get("section_river_center_m", ""),
        "最近断面河中X": nearest["section"].get("section_river_center_x", ""),
        "最近断面河中Y": nearest["section"].get("section_river_center_y", ""),
        "最近断面是否使用河中坐标锚定": "是" if nearest["section"].get("section_table_anchor_used", False) else "否",
        "最近断面河中坐标到断面线距离_m": nearest["section"].get("section_table_anchor_distance_m", ""),
        "校正后河道路径": str(corrected_river_path) if corrected_river_path else "",
        "最近断面裁剪起点类型": nearest["section"].get("section_trim_start_label", ""),
        "最近断面裁剪终点类型": nearest["section"].get("section_trim_end_label", ""),
        "最近断面原本深度_m": nearest["section"].get("section_original_depth_m", ""),
        "最近断面河中高程_m": nearest["section"].get("section_riverbed_elevation_m", ""),
        "最近断面两端较低边界高程_m": nearest["section"].get("section_boundary_elevation_m", ""),
        "断面值类型": value_label,
        f"最近断面在首次受淹时刻的{value_label}": current_section_value,
        "用户输入的前序时间间隔": interval_hours,
        "前序时刻": previous_label or "",
        f"最近断面在前序时刻的{value_label}": previous_section_value,
        "断面取值方法": section_method,
        "备注": " ".join(note_parts) if note_parts else f"建筑物筛选方式：{filter_method}；最近断面按参考对象到所有断面的最短直线距离选择。",
    }

    xlsx_path, csv_path = write_outputs(summary, building_details, flooded_details, output_dir, logger, section_distances=section_distances)
    logger.info(f"Excel 输出：{xlsx_path}")
    logger.info(f"CSV 输出：{csv_path}")
    return AnalysisResult(summary, building_details, flooded_details, section_distances, map_payload, xlsx_path, csv_path)
