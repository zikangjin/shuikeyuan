from __future__ import annotations

import argparse
import json
import math
import mimetypes
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import geopandas as gpd
import rasterio
import yaml
from shapely.geometry import LineString

from src.data_loader import (
    VILLAGE_FIELD_CANDIDATES,
    build_where,
    first_existing_field,
    load_section_table_trim_ranges,
    load_sections,
    scan_flood_rasters,
    read_vector,
    resolve_vector_path,
)
from src.flood_analysis import (
    _measured_centerlines_from_sections,
    _raster_crs_from_config,
    _raster_max_value,
    _section_order_value,
    _trim_sections_for_context,
    _union_geometry,
    run_analysis,
)
from src.logger_utils import WorkflowLogger
from src.raster_utils import crop_raster_to_temp, section_value_stats


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "web_static"
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
RIVER_QUERIES: dict[str, dict] = {}
RIVER_QUERIES_LOCK = threading.Lock()


WEB_CONFIG_KEYS = [
    "building_path",
    "flood_folder",
    "section_path",
    "auxiliary_path",
    "river_network_path",
    "output_dir",
    "village_name",
    "threshold",
    "time_interval_hours",
    "value_type",
    "scenario_name",
    "section_reference_mode",
    "show_river_network",
    "export_corrected_river_network",
    "river_query_village_buffer_m",
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def coerce_value(key: str, value):
    if value == "":
        return None
    if key in {"threshold", "time_interval_hours", "river_query_village_buffer_m"}:
        return float(value)
    if key in {"show_river_network", "export_corrected_river_network"}:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "是", "y"}
    return value


def clean_config_for_web(config: dict) -> dict:
    return {key: config.get(key) for key in WEB_CONFIG_KEYS}


def json_response(handler: BaseHTTPRequestHandler, data, status: int = 200) -> None:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler: BaseHTTPRequestHandler, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def file_response(handler: BaseHTTPRequestHandler, path: Path) -> None:
    if not path.exists() or not path.is_file():
        text_response(handler, "Not found", status=404)
        return
    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    body = path.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def gdf_to_geojson(gdf):
    if gdf is None or getattr(gdf, "empty", True):
        return {"type": "FeatureCollection", "features": []}
    return json.loads(gdf.to_json())


def normalize_text(value) -> str:
    return "".join(str(value or "").split()).lower()


def compact_unique(values, limit: int = 200) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(text)
        if len(rows) >= limit:
            break
    return rows


def safe_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def available_river_names(sections: gpd.GeoDataFrame) -> list[str]:
    values: list[str] = []
    for field in ("river_folder", "section_name", "section_id"):
        if field in sections.columns:
            values.extend(sections[field].dropna().astype(str).tolist())
    names: list[str] = []
    for value in compact_unique(values, limit=1000):
        if "_" in value:
            names.append(value.split("_")[0])
        else:
            names.append(value)
    return compact_unique(names, limit=200)


def filter_sections_by_river(sections: gpd.GeoDataFrame, river_name: str) -> gpd.GeoDataFrame:
    query = normalize_text(river_name)
    if not query:
        raise ValueError("请输入沟道名称。")
    masks = []
    for field in ("river_folder", "section_name", "section_id", "source_path"):
        if field not in sections.columns:
            continue
        values = sections[field].fillna("").astype(str).map(normalize_text)
        masks.append(values.str.contains(query, regex=False))
    if not masks:
        return sections.iloc[0:0].copy()
    mask = masks[0]
    for one in masks[1:]:
        mask = mask | one
    return sections[mask].copy()


def section_columns_for_web(sections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    wanted = [
        "section_id",
        "section_name",
        "river_folder",
        "source_path",
        "point_count",
        "original_length_m",
        "trimmed_length_m",
        "section_trim_method",
        "section_original_depth_m",
        "section_river_center_m",
        "section_trim_start_label",
        "section_trim_end_label",
        "geometry",
    ]
    return sections[[col for col in wanted if col in sections.columns]].copy()


def load_all_village_boundaries(config: dict) -> tuple[gpd.GeoDataFrame | None, str | None]:
    auxiliary_path = str(config.get("auxiliary_path") or "").strip()
    if not auxiliary_path:
        return None, None
    boundary = read_vector(
        Path(auxiliary_path),
        crs_if_missing=config.get("auxiliary_crs"),
        encoding=config.get("auxiliary_encoding") or "gbk",
    )
    if boundary.empty:
        return None, None
    if boundary.crs is None:
        raise ValueError("村庄边界缺少 CRS，无法判断沟道涉及村庄。")
    configured = str(config.get("auxiliary_village_field") or "").strip()
    village_field = configured if configured in boundary.columns else first_existing_field(boundary.columns, VILLAGE_FIELD_CANDIDATES)
    if village_field is None:
        raise ValueError(f"村庄边界中无法识别村名字段。可用字段：{', '.join(map(str, boundary.columns))}")
    return boundary, village_field


def load_river_query_buildings(config: dict, villages_gdf, query_geom, target_crs, logger) -> tuple[gpd.GeoDataFrame | None, str | None]:
    building_path = str(config.get("building_path") or "").strip()
    if not building_path:
        return None, None
    path = Path(building_path)
    if not path.exists():
        logger.warning(f"建筑物数据不存在，沟道查询不显示建筑物：{path}")
        return None, None

    clip_geom = None
    if villages_gdf is not None and not getattr(villages_gdf, "empty", True):
        clip_geom = _union_geometry(villages_gdf.to_crs(target_crs))
    elif query_geom is not None and not query_geom.is_empty:
        clip_geom = query_geom

    read_bbox = None
    if clip_geom is not None and not clip_geom.is_empty:
        try:
            source_path = resolve_vector_path(path)
            allow_bbox = bool(config.get("river_query_use_building_bbox")) or source_path.suffix.lower() == ".gpkg"
            if allow_bbox:
                source_crs = config.get("building_crs")
                if not source_crs and source_path.suffix.lower() in {".shp", ".geojson", ".json", ".gpkg"}:
                    kwargs = {}
                    if config.get("building_encoding"):
                        kwargs["encoding"] = config.get("building_encoding")
                    header = gpd.read_file(source_path, rows=0, **kwargs)
                    source_crs = header.crs
                if source_crs:
                    bbox_geom = gpd.GeoDataFrame(geometry=[clip_geom], crs=target_crs).to_crs(source_crs).geometry.iloc[0]
                    read_bbox = tuple(float(value) for value in bbox_geom.bounds)
                    logger.info("沟道查询按空间范围读取建筑物，避免加载全市白膜。")
            else:
                logger.info("当前建筑物为 Shapefile，跳过 bbox 读取以避免驱动全表扫描。")
        except Exception as exc:
            logger.warning(f"建筑物空间范围预筛失败，将使用常规读取：{exc}")

    pre_field = str(config.get("building_prefilter_field") or "").strip()
    pre_value = str(config.get("building_prefilter_value") or "").strip()
    where = build_where(pre_field, pre_value) if pre_field and pre_value else None
    if where:
        logger.info(f"沟道查询按属性预筛建筑物：{pre_field} = {pre_value}")
    buildings = read_vector(
        path,
        crs_if_missing=config.get("building_crs"),
        encoding=config.get("building_encoding"),
        where=where,
        bbox=read_bbox,
    )
    if buildings.empty:
        logger.warning("建筑物数据为空，沟道查询不显示建筑物。")
        return None, None
    if buildings.crs is None:
        logger.warning("建筑物数据缺少 CRS，沟道查询不显示建筑物。")
        return None, None

    buildings_proj = buildings.to_crs(target_crs)
    if clip_geom is not None and not clip_geom.is_empty:
        minx, miny, maxx, maxy = clip_geom.bounds
        candidates = buildings_proj.cx[minx:maxx, miny:maxy].copy()
        buildings_proj = candidates[candidates.intersects(clip_geom)].copy() if not candidates.empty else candidates

    if buildings_proj.empty:
        logger.warning("沟道涉及范围内没有匹配到建筑物。")
        return buildings_proj, None
    id_field = str(config.get("building_id_field") or "").strip()
    if not id_field or id_field not in buildings_proj.columns:
        for candidate in ("OBJECTID", "FID", "id", "ID", "Name", "name"):
            if candidate in buildings_proj.columns:
                id_field = candidate
                break
    if not id_field or id_field not in buildings_proj.columns:
        id_field = "building_id"
        buildings_proj = buildings_proj.copy()
        buildings_proj[id_field] = [str(i + 1) for i in range(len(buildings_proj))]
    logger.info(f"沟道查询建筑物数量：{len(buildings_proj)}")
    return buildings_proj, id_field

def combined_scope_gdf(layers: list[gpd.GeoDataFrame | None], crs) -> gpd.GeoDataFrame:
    geometries = []
    for layer in layers:
        if layer is None or getattr(layer, "empty", True):
            continue
        for geom in layer.to_crs(crs).geometry:
            if geom is not None and not geom.is_empty:
                geometries.append(geom)
    return gpd.GeoDataFrame(geometry=geometries, crs=crs)


def build_river_query_frames(config: dict, scope_gdf: gpd.GeoDataFrame, logger) -> tuple[list[dict], object | None, dict]:
    flood_folder = str(config.get("flood_folder") or "").strip()
    if not flood_folder or scope_gdf is None or scope_gdf.empty:
        return [], None, {}
    rasters = scan_flood_rasters(Path(flood_folder), str(config.get("value_type") or "水深"), logger)
    raster_crs = _raster_crs_from_config(rasters[0].path, config, scope_gdf, logger)
    scope_raster = scope_gdf.to_crs(raster_crs)
    bounds = tuple(float(value) for value in scope_raster.total_bounds)
    scenario = str(config.get("scenario_name") or Path(flood_folder).name or "scenario")
    temp_base = Path(config.get("temp_dir") or Path(config.get("output_dir") or "outputs") / "temp")
    target = temp_base / scenario / "river_query"
    padding = int(config.get("raster_crop_padding_cells") or 2)
    frames = [
        {
            "index": idx,
            "label": item.label,
            "raster_path": str(item.path),
            "max_value": None,
        }
        for idx, item in enumerate(rasters)
    ]
    logger.info(f"沟道查询动画帧数量：{len(frames)}；淹没栅格将在播放到对应时刻时加载。")
    return frames, raster_crs, {
        "crop_bounds": bounds,
        "crop_target": str(target),
        "raster_crs": raster_crs,
        "crop_padding": padding,
    }

def trim_sections_for_river_query(sections: gpd.GeoDataFrame, config: dict, logger) -> gpd.GeoDataFrame:
    analysis_crs = sections.crs
    query_config = dict(config)
    query_config["section_avoid_buildings"] = False
    try:
        trimmed = _trim_sections_for_context(
            sections,
            query_config,
            analysis_crs,
            sections,
            None,
            None,
            None,
            load_section_table_trim_ranges(query_config, logger),
            logger,
            avoid_buildings_gdf=None,
        )
        valid = trimmed.geometry.apply(lambda geom: geom is not None and not geom.is_empty)
        if valid.any():
            return trimmed[valid].copy()
    except Exception as exc:
        logger.warning(f"沟道查询使用原始断面线显示，原因：{exc}")
    result = sections.to_crs(analysis_crs).copy()
    result["original_length_m"] = result.geometry.length
    result["trimmed_length_m"] = result.geometry.length
    result["section_trim_method"] = "原始断面"
    return result


def centerline_for_river_query(sections: gpd.GeoDataFrame):
    centerlines = _measured_centerlines_from_sections(sections, sections.crs)
    if centerlines is not None and not centerlines.empty:
        return centerlines

    points: list[tuple[int, float, float, str]] = []
    for _, row in sections.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        midpoint = geom.interpolate(0.5, normalized=True)
        section_id = str(row.get("section_id", ""))
        points.append((_section_order_value(section_id), float(midpoint.x), float(midpoint.y), section_id))
    points.sort(key=lambda item: item[0])
    unique_points: list[tuple[float, float]] = []
    section_ids: list[str] = []
    for _order, x, y, section_id in points:
        xy = (x, y)
        if not unique_points or xy != unique_points[-1]:
            unique_points.append(xy)
        section_ids.append(section_id)
    if len(unique_points) < 2:
        return None
    river_name = str(sections.iloc[0].get("river_folder", "") or "查询沟道")
    return gpd.GeoDataFrame(
        [
            {
                "river_name": river_name,
                "source": "section_midpoints",
                "section_ids": "、".join(section_ids[:50]),
                "point_count": len(unique_points),
                "geometry": LineString(unique_points),
            }
        ],
        geometry="geometry",
        crs=sections.crs,
    )


def river_query(config: dict, river_name: str) -> dict:
    logs: list[str] = []
    logger = WorkflowLogger(callback=logs.append, log_file=None)
    sections, _section_id_field, _section_name_field = load_sections(config, logger)
    filtered = filter_sections_by_river(sections, river_name)
    if filtered.empty:
        return {
            "query": river_name,
            "error": f"没有找到名称包含“{river_name}”的沟道断面。",
            "available_rivers": available_river_names(sections)[:60],
            "logs": logs,
        }

    trimmed = trim_sections_for_river_query(filtered, config, logger)
    trimmed = trimmed.sort_values("section_id", key=lambda s: s.map(_section_order_value)).copy()
    river_line = centerline_for_river_query(trimmed)
    query_geom = _union_geometry(river_line) if river_line is not None and not river_line.empty else _union_geometry(trimmed)

    buffer_m = float(config.get("river_query_village_buffer_m") or 150)
    villages_map = None
    village_rows: list[dict] = []
    village_field = None
    if query_geom is not None and not query_geom.is_empty:
        boundaries, village_field = load_all_village_boundaries(config)
        if boundaries is not None and village_field is not None:
            boundaries_proj = boundaries.to_crs(trimmed.crs)
            area = query_geom.buffer(buffer_m)
            candidates = boundaries_proj[boundaries_proj.intersects(area)].copy()
            if not candidates.empty:
                candidates["village_name"] = candidates[village_field].astype(str)
                villages_map = candidates[["village_name", "geometry"]].copy()
                for _, row in candidates.sort_values("village_name").iterrows():
                    village_rows.append({"village_name": str(row["village_name"])})

    query_area = query_geom.buffer(buffer_m) if query_geom is not None and not query_geom.is_empty else None
    buildings_map, building_id_field = load_river_query_buildings(config, villages_map, query_area, trimmed.crs, logger)
    scope_gdf = combined_scope_gdf([buildings_map, villages_map, trimmed, river_line], trimmed.crs)
    frames, raster_crs, frame_options = build_river_query_frames(config, scope_gdf, logger)
    map_crs = raster_crs or trimmed.crs
    trimmed_web = section_columns_for_web(trimmed.to_crs(map_crs))
    river_web = river_line.to_crs(map_crs) if river_line is not None and not river_line.empty else river_line
    villages_web = villages_map.to_crs(map_crs) if villages_map is not None and not villages_map.empty else villages_map
    buildings_web = None
    if buildings_map is not None and not buildings_map.empty:
        buildings_for_web = buildings_map.to_crs(map_crs)
        if building_id_field and building_id_field in buildings_for_web.columns:
            buildings_web = buildings_for_web[[building_id_field, "geometry"]].copy()
        else:
            buildings_web = buildings_for_web[["geometry"]].copy()
    section_rows = []
    for _, row in trimmed.iterrows():
        section_rows.append(
            {
                "section_id": str(row.get("section_id", "")),
                "section_name": str(row.get("section_name", "")),
                "river_folder": str(row.get("river_folder", "")),
                "length_m": float(row.geometry.length) if row.geometry is not None and not row.geometry.is_empty else None,
                "original_depth_m": safe_float(row.get("section_original_depth_m", None)),
                "trim_method": str(row.get("section_trim_method", "")),
            }
        )

    matched_river = str(trimmed.iloc[0].get("river_folder", "") or river_name)
    return {
        "query": river_name,
        "matched_river": matched_river,
        "section_count": len(trimmed),
        "village_count": len(village_rows),
        "village_buffer_m": buffer_m,
        "village_field": village_field,
        "villages": village_rows,
        "sections": section_rows,
        "logs": logs,
        "map": {
            "sections": gdf_to_geojson(trimmed_web),
            "river": gdf_to_geojson(river_web),
            "villages": gdf_to_geojson(villages_web),
            "buildings": gdf_to_geojson(buildings_web),
            "frames": [{"index": frame["index"], "label": frame["label"], "max_value": frame.get("max_value")} for frame in frames],
            "threshold": float(config.get("threshold") or 0),
            "value_label": "水深" if "位" not in str(config.get("value_type") or "水深") else "水位",
        },
        "_internal": {
            "frames": frames,
            "sections": trimmed_web,
            "river": river_web,
            "threshold": float(config.get("threshold") or 0),
            "section_buffer_m": float(config.get("section_buffer_m") or 5),
            "frame_cache": {},
            **frame_options,
        },
    }


def serialize_result(result) -> dict:
    payload = result.map_payload or {}
    frames = []
    for idx, frame in enumerate(payload.get("frames") or []):
        frames.append(
            {
                "index": idx,
                "label": frame.get("label"),
                "flooded_count": frame.get("flooded_count"),
                "max_value": frame.get("max_value"),
                "threshold_area_ratio_pct": frame.get("threshold_area_ratio_pct"),
            }
        )
    return {
        "summary": result.summary,
        "xlsx_path": str(result.xlsx_path),
        "csv_path": str(result.csv_path),
        "map": {
            "buildings": gdf_to_geojson(payload.get("buildings")),
            "reference_buildings": gdf_to_geojson(payload.get("reference_buildings")),
            "sections": gdf_to_geojson(payload.get("sections")),
            "rivers": gdf_to_geojson(payload.get("rivers")),
            "corrected_rivers": gdf_to_geojson(payload.get("corrected_rivers")),
            "frames": frames,
            "threshold": payload.get("threshold"),
            "value_label": payload.get("value_label"),
            "village_name": payload.get("village_name"),
            "scenario_name": payload.get("scenario_name"),
            "nearest_section_id": payload.get("nearest_section_id"),
            "first_flood_time": payload.get("first_flood_time"),
        },
    }


def raster_cells(path: Path, threshold: float, max_grid: int = 170) -> dict:
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
        transform = src.transform
    if data.size == 0:
        return {"cells": []}
    height, width = data.shape
    step = max(1, int(math.ceil(max(height, width) / max_grid)))
    cells = []
    min_value = max(float(threshold) * 0.25, 0.02) if threshold else 0.02
    max_value = None
    for row in range(0, height, step):
        for col in range(0, width, step):
            block = data[row : min(row + step, height), col : min(col + step, width)]
            if np.ma.is_masked(block):
                if block.mask.all():
                    continue
                value = float(block.max())
            else:
                value = float(np.nanmax(block))
            if not math.isfinite(value) or value <= 0 or value < min_value:
                continue
            x0, y0 = transform * (col, row)
            x1, y1 = transform * (min(col + step, width), min(row + step, height))
            cells.append([x0, y0, x1, y1, value])
            max_value = value if max_value is None else max(max_value, value)
    return {"cells": cells, "max_value": max_value, "step": step}


def live_geometry_values(result, raster_path: Path) -> dict:
    payload = result.map_payload or {}
    buffer_m = float(payload.get("section_buffer_m") or 5)
    values: dict[str, dict] = {}
    river_values: dict[str, dict] = {}

    sections = payload.get("sections")
    if sections is not None and not getattr(sections, "empty", True):
        for idx, row in sections.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            section_id = str(row.get("section_id", idx))
            try:
                stats = section_value_stats(geom, raster_path, method="web_frame_section", buffer_m=buffer_m)
            except Exception:
                stats = {"max": None, "mean": None, "median": None}
            values[section_id] = stats

    corrected_rivers = payload.get("corrected_rivers")
    if corrected_rivers is not None and not getattr(corrected_rivers, "empty", True):
        for idx, row in corrected_rivers.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            river_id = str(row.get("river_name", "") or row.get("river_folder", "") or idx)
            try:
                stats = section_value_stats(geom, raster_path, method="web_frame_corrected_river", buffer_m=buffer_m)
            except Exception:
                stats = {"max": None, "mean": None, "median": None}
            river_values[river_id] = stats

    return {"section_values": values, "corrected_river_values": river_values}


def live_river_query_values(cache: dict, raster_path: Path) -> dict:
    buffer_m = float(cache.get("section_buffer_m") or 5)
    values: dict[str, dict] = {}
    river_values: dict[str, dict] = {}
    sections = cache.get("sections")
    if sections is not None and not getattr(sections, "empty", True):
        for idx, row in sections.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            section_id = str(row.get("section_id", idx))
            try:
                stats = section_value_stats(geom, raster_path, method="river_query_section", buffer_m=buffer_m)
            except Exception:
                stats = {"max": None, "mean": None, "median": None}
            values[section_id] = stats
    river = cache.get("river")
    if river is not None and not getattr(river, "empty", True):
        for idx, row in river.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            river_id = str(row.get("river_name", idx))
            try:
                stats = section_value_stats(geom, raster_path, method="river_query_river", buffer_m=buffer_m)
            except Exception:
                stats = {"max": None, "mean": None, "median": None}
            river_values[river_id] = stats
    return {"section_values": values, "corrected_river_values": river_values}

def run_job(job_id: str, config: dict) -> None:
    def log(line: str) -> None:
        with JOBS_LOCK:
            JOBS[job_id]["logs"].append(line)
            JOBS[job_id]["logs"] = JOBS[job_id]["logs"][-300:]

    logger = WorkflowLogger(callback=log, log_file=Path(config.get("output_dir") or "outputs") / "logs" / "web_workflow.log")
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["started_at"] = time.time()
    try:
        result = run_analysis(config, logger)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["result"] = result
            JOBS[job_id]["serialized"] = serialize_result(result)
            JOBS[job_id]["finished_at"] = time.time()
    except Exception as exc:
        logger.error(str(exc))
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(exc)
            JOBS[job_id]["finished_at"] = time.time()


class WebHandler(BaseHTTPRequestHandler):
    config_path = ROOT / "config.yaml"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            file_response(self, STATIC_DIR / "index.html")
            return
        if path.startswith("/static/"):
            file_response(self, STATIC_DIR / path.removeprefix("/static/"))
            return
        if path == "/api/config":
            config = load_config(self.config_path)
            json_response(self, clean_config_for_web(config))
            return
        if path.startswith("/api/river-query/"):
            parts = [part for part in path.split("/") if part]
            if len(parts) == 5 and parts[3] == "frame":
                self._river_query_frame(parts[2], int(parts[4]))
                return
        if path.startswith("/api/job/"):
            parts = [part for part in path.split("/") if part]
            if len(parts) == 3:
                self._job_status(parts[2])
                return
            if len(parts) == 5 and parts[3] == "frame":
                self._frame_cells(parts[2], int(parts[4]))
                return
        text_response(self, "Not found", status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/river-query":
            self._river_query()
            return
        if parsed.path != "/api/run":
            text_response(self, "Not found", status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        incoming = json.loads(body or "{}")
        config = load_config(self.config_path)
        for key, value in incoming.items():
            if key in WEB_CONFIG_KEYS:
                config[key] = coerce_value(key, value)
        job_id = uuid.uuid4().hex[:12]
        with JOBS_LOCK:
            JOBS[job_id] = {"status": "queued", "logs": [], "frame_cache": {}, "created_at": time.time()}
        thread = threading.Thread(target=run_job, args=(job_id, config), daemon=True)
        thread.start()
        json_response(self, {"job_id": job_id, "status": "queued"})

    def _river_query(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        incoming = json.loads(body or "{}")
        config = load_config(self.config_path)
        for key, value in incoming.items():
            if key in WEB_CONFIG_KEYS:
                config[key] = coerce_value(key, value)
        river_name = str(incoming.get("river_name") or incoming.get("river_query_name") or "").strip()
        try:
            data = river_query(config, river_name)
            internal = data.pop("_internal", None)
            if internal is not None and not data.get("error"):
                query_id = uuid.uuid4().hex[:12]
                with RIVER_QUERIES_LOCK:
                    RIVER_QUERIES[query_id] = internal
                data["query_id"] = query_id
            status = 404 if data.get("error") else 200
            json_response(self, data, status=status)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, status=500)

    def _river_query_frame(self, query_id: str, index: int) -> None:
        with RIVER_QUERIES_LOCK:
            cache = RIVER_QUERIES.get(query_id)
            if not cache:
                json_response(self, {"error": "river query not found"}, status=404)
                return
            frame_cache = cache.setdefault("frame_cache", {})
            if index in frame_cache:
                json_response(self, frame_cache[index])
                return
            frames = cache.get("frames") or []
            threshold = float(cache.get("threshold") or 0)
            if index < 0 or index >= len(frames):
                json_response(self, {"error": "frame out of range"}, status=404)
                return
            frame = frames[index]
            source_path = Path(frame.get("raster_path") or "")
            cropped_path = frame.get("cropped_raster_path")
            crop_bounds = cache.get("crop_bounds")
            crop_target = cache.get("crop_target")
            raster_crs = cache.get("raster_crs")
            crop_padding = int(cache.get("crop_padding") or 2)

        try:
            if cropped_path:
                raster_path = Path(cropped_path)
            elif crop_bounds and crop_target:
                raster_path = crop_raster_to_temp(
                    source_path,
                    tuple(crop_bounds),
                    Path(crop_target),
                    raster_crs=raster_crs,
                    padding_pixels=crop_padding,
                )
                with RIVER_QUERIES_LOCK:
                    current = RIVER_QUERIES.get(query_id)
                    if current is not None:
                        current_frames = current.get("frames") or []
                        if 0 <= index < len(current_frames):
                            current_frames[index]["cropped_raster_path"] = str(raster_path)
            else:
                raster_path = source_path
            data = raster_cells(raster_path, threshold)
            data.update(live_river_query_values(cache, raster_path))
            data["label"] = frame.get("label")
            if data.get("max_value") is not None:
                with RIVER_QUERIES_LOCK:
                    current = RIVER_QUERIES.get(query_id)
                    if current is not None:
                        current_frames = current.get("frames") or []
                        if 0 <= index < len(current_frames):
                            current_frames[index]["max_value"] = data.get("max_value")
        except Exception as exc:
            json_response(self, {"error": str(exc)}, status=500)
            return

        with RIVER_QUERIES_LOCK:
            cache = RIVER_QUERIES.get(query_id)
            if cache is not None:
                cache.setdefault("frame_cache", {})[index] = data
        json_response(self, data)
    def _job_status(self, job_id: str) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                json_response(self, {"error": "job not found"}, status=404)
                return
            data = {
                "job_id": job_id,
                "status": job.get("status"),
                "logs": job.get("logs", [])[-80:],
                "error": job.get("error"),
                "result": job.get("serialized") if job.get("status") == "done" else None,
            }
        json_response(self, data)

    def _frame_cells(self, job_id: str, index: int) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job or job.get("status") != "done":
                json_response(self, {"error": "job not ready"}, status=404)
                return
            if index in job["frame_cache"]:
                json_response(self, job["frame_cache"][index])
                return
            result = job["result"]
            frames = (result.map_payload or {}).get("frames") or []
            threshold = float((result.map_payload or {}).get("threshold") or 0)
            if index < 0 or index >= len(frames):
                json_response(self, {"error": "frame out of range"}, status=404)
                return
            raster_path = Path(frames[index]["raster_path"])
        data = raster_cells(raster_path, threshold)
        data.update(live_geometry_values(result, raster_path))
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job["frame_cache"][index] = data
        json_response(self, data)

    def log_message(self, _format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="村庄淹没分析 Web 控制台")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8765, help="监听端口")
    args = parser.parse_args()
    WebHandler.config_path = Path(args.config)
    server = ThreadingHTTPServer((args.host, args.port), WebHandler)
    print(f"Web app running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

















