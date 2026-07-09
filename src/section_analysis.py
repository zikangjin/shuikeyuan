from __future__ import annotations

import geopandas as gpd
import pandas as pd
from pyproj import CRS


def choose_analysis_crs(buildings: gpd.GeoDataFrame, raster_crs, target_projected_crs: str | None, logger):
    if target_projected_crs:
        return CRS.from_user_input(target_projected_crs)
    crs = CRS.from_user_input(raster_crs) if raster_crs else CRS.from_user_input(buildings.crs)
    if not crs.is_geographic:
        return crs
    estimated = buildings.estimate_utm_crs()
    if estimated:
        logger.warning(f"输入坐标系是经纬度，距离计算自动改用估算投影坐标系：{estimated}")
        return estimated
    logger.warning("输入坐标系是经纬度，无法估算 UTM，距离计算临时使用 EPSG:3857。")
    return CRS.from_epsg(3857)


def union_geometry(gdf: gpd.GeoDataFrame):
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union


def nearest_section(
    reference_gdf: gpd.GeoDataFrame,
    sections: gpd.GeoDataFrame,
    analysis_crs,
    id_field: str,
    name_field: str,
    reference_type: str,
):
    reference_projected = reference_gdf.to_crs(analysis_crs)
    sections_projected = sections.to_crs(analysis_crs)
    valid_geometry = sections_projected.geometry.apply(lambda geom: geom is not None and not geom.is_empty)
    sections_projected = sections_projected[valid_geometry].copy()
    if sections_projected.empty:
        raise ValueError("没有可用的裁剪后断面。请检查道路/路网、河流水系与断面是否相交。")
    reference_geom = union_geometry(reference_projected)
    distances = sections_projected.geometry.distance(reference_geom)
    idx = distances.idxmin()
    row = sections_projected.loc[idx]
    sorted_distances = distances.sort_values()
    rows = []
    for rank, (section_idx, distance) in enumerate(sorted_distances.items(), start=1):
        section = sections_projected.loc[section_idx]
        rows.append(
            {
                "rank": rank,
                "section_index": section_idx,
                "section_id": str(section.get(id_field, section_idx)),
                "section_name": str(section.get(name_field, section.get(id_field, section_idx))),
                "distance_m": float(distance),
                "reference_type": reference_type,
                "geometry_type": section.geometry.geom_type,
                "source_path": section.get("source_path", ""),
                "river_folder": section.get("river_folder", ""),
                "point_count": section.get("point_count", ""),
                "original_length_m": section.get("original_length_m", ""),
                "trimmed_length_m": section.get("trimmed_length_m", ""),
                "section_trim_method": section.get("section_trim_method", ""),
                "section_trim_start_m": section.get("section_trim_start_m", ""),
                "section_trim_end_m": section.get("section_trim_end_m", ""),
                "section_river_center_m": section.get("section_river_center_m", ""),
                "section_trim_start_label": section.get("section_trim_start_label", ""),
                "section_trim_end_label": section.get("section_trim_end_label", ""),
                "section_river_center_x": section.get("section_river_center_x", ""),
                "section_river_center_y": section.get("section_river_center_y", ""),
                "section_table_anchor_used": section.get("section_table_anchor_used", ""),
                "section_table_anchor_distance_m": section.get("section_table_anchor_distance_m", ""),
                "section_river_center_analysis_x": section.get("section_river_center_analysis_x", ""),
                "section_river_center_analysis_y": section.get("section_river_center_analysis_y", ""),
                "section_original_depth_m": section.get("section_original_depth_m", ""),
                "section_riverbed_elevation_m": section.get("section_riverbed_elevation_m", ""),
                "section_boundary_elevation_m": section.get("section_boundary_elevation_m", ""),
            }
        )
    return {
        "section": row,
        "geometry_analysis_crs": row.geometry,
        "reference_geometry": reference_geom,
        "reference_type": reference_type,
        "distance_m": float(distances.loc[idx]),
        "section_id": str(row.get(id_field, idx)),
        "section_name": str(row.get(name_field, row.get(id_field, idx))),
        "section_distances": pd.DataFrame(rows),
    }
