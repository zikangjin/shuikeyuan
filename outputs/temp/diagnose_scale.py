from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_loader import (
    load_buildings_for_village,
    load_river_network,
    load_sections,
    load_village_boundary,
    scan_flood_rasters,
)
from src.flood_analysis import _raster_crs_from_config
from src.logger_utils import WorkflowLogger


def wh(gdf):
    minx, miny, maxx, maxy = gdf.total_bounds
    return round(maxx - minx, 2), round(maxy - miny, 2), tuple(round(x, 2) for x in gdf.total_bounds)


def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    logger = WorkflowLogger()

    buildings, _village_field, _filter_method = load_buildings_for_village(cfg, logger)
    rasters = scan_flood_rasters(Path(cfg["flood_folder"]), cfg.get("value_type", "水深"), logger)
    raster_crs = _raster_crs_from_config(rasters[0].path, cfg, buildings, logger)
    cfg["raster_crs"] = raster_crs.to_string()

    buildings_raster = buildings.to_crs(raster_crs)
    boundary = load_village_boundary(
        Path(cfg["auxiliary_path"]),
        cfg["village_name"],
        field=cfg.get("auxiliary_village_field"),
        encoding=cfg.get("auxiliary_encoding") or "gbk",
    ).to_crs(raster_crs)
    sections, _sid, _sname = load_sections(cfg, logger)
    sections_raster = sections.to_crs(raster_crs)
    rivers = load_river_network(cfg, logger).to_crs(raster_crs)

    with rasterio.open(rasters[0].path) as src:
        embedded_raster_crs = src.crs
        raster_res = src.res
        raster_bounds = src.bounds
        raster_nodata = src.nodata

    print("\n--- CRS ---")
    print("buildings_crs:", buildings.crs)
    print("raster_embedded_crs:", embedded_raster_crs)
    print("raster_assumed_crs:", raster_crs)
    print("sections_crs_after:", sections_raster.crs)
    print("rivers_crs_after:", rivers.crs)

    print("\n--- Bounds / size in raster CRS ---")
    print("village_boundary_w_h_bounds:", wh(boundary))
    print("selected_buildings_w_h_bounds:", wh(buildings_raster))
    print("sections_all_w_h_bounds:", wh(sections_raster))
    print("rivers_all_w_h_bounds:", wh(rivers))
    print("raster_bounds:", tuple(round(x, 2) for x in raster_bounds))
    print("raster_res:", raster_res)
    print("raster_nodata:", raster_nodata)
    print("village_area_m2:", round(float(boundary.geometry.area.sum()), 2))
    print("buildings_count:", len(buildings_raster))
    print("buildings_area_m2:", round(float(buildings_raster.geometry.area.sum()), 2))

    print("\n--- Section length diagnostics ---")
    sections_raster = sections_raster.copy()
    sections_raster["length_m"] = sections_raster.geometry.length
    sections_raster["dist_to_village_m"] = sections_raster.geometry.distance(boundary.geometry.union_all())
    xmin, ymin, xmax, ymax = boundary.total_bounds
    pad = 2000
    near = sections_raster.cx[xmin - pad : xmax + pad, ymin - pad : ymax + pad].copy()
    near = near.sort_values("dist_to_village_m")
    print("near_sections_count_2km_bbox:", len(near))
    cols = ["section_id", "section_name", "river_folder", "length_m", "dist_to_village_m", "source_path"]
    print(near[cols].head(12).to_string(index=False))
    print("\nlongest_sections_overall:")
    print(sections_raster.sort_values("length_m", ascending=False)[cols[:-1] + ["source_path"]].head(8).to_string(index=False))
    print("\nnearest_sections_overall:")
    print(sections_raster.sort_values("dist_to_village_m")[cols].head(12).to_string(index=False))

    print("\n--- Raster village mask ratios ---")
    geoms = [boundary.geometry.union_all().__geo_interface__]
    rows = []
    for hour in [1, 2, 3, 4, 6, 12, 24]:
        matches = [item for item in rasters if int(item.hour_value or -1) == hour]
        if not matches:
            continue
        item = matches[0]
        with rasterio.open(item.path) as src:
            minx, miny, maxx, maxy = boundary.total_bounds
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            window = window.round_offsets().round_lengths()
            data = src.read(1, window=window)
            transform = src.window_transform(window)
            inside = geometry_mask(geoms, out_shape=data.shape, transform=transform, invert=True)
        all_inside = data[inside]
        if all_inside.size == 0:
            rows.append({"time": item.label, "valid_cells": 0})
            continue
        nonzero = all_inside[all_inside > 0]
        comp = nonzero if nonzero.size else np.array([0.0])
        rows.append(
            {
                "time": item.label,
                "inside_cells": int(all_inside.size),
                "nonzero_cells": int(nonzero.size),
                "min": float(comp.min()),
                "max": float(comp.max()),
                "mean": float(comp.mean()),
                "pct_gt_0_of_village": float((all_inside > 0).mean() * 100),
                "pct_ge_0_05_of_village": float((all_inside >= 0.05).mean() * 100),
                "pct_ge_0_2_of_village": float((all_inside >= 0.2).mean() * 100),
                "cells_ge_0_2": int((all_inside >= 0.2).sum()),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n--- River overlap near village ---")
    minx, miny, maxx, maxy = boundary.total_bounds
    river_near = rivers.cx[minx - 2000 : maxx + 2000, miny - 2000 : maxy + 2000].copy()
    river_near = river_near[river_near.intersects(box(minx - 2000, miny - 2000, maxx + 2000, maxy + 2000))]
    print("river_near_count_2km_bbox:", len(river_near))
    if not river_near.empty:
        river_near["dist_to_village_m"] = river_near.geometry.distance(boundary.geometry.union_all())
        cols = [c for c in ["NAME", "TYPE_", "Shape_Leng", "dist_to_village_m"] if c in river_near.columns]
        print(river_near.sort_values("dist_to_village_m")[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
