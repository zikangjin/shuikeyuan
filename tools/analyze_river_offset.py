from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.affinity import translate
from shapely.geometry import LineString
from shapely.ops import nearest_points


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RIVER_NAME_FIELDS = [
    "river_name",
    "河道名称",
    "河流名称",
    "河流名",
    "河名",
    "沟道名称",
    "沟名",
    "水系名称",
    "名称",
    "NAME",
    "Name",
    "name",
    "RIVER",
    "River",
    "river",
    "RIV_NAME",
    "HYNAME",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="诊断原始河道与实测河中点之间的偏移规律。")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径，默认 config.yaml")
    parser.add_argument("--river-path", default=None, help="原始河流水系路径；不填则读取 config.yaml 的 river_network_path")
    parser.add_argument(
        "--measured-points",
        default=None,
        help="实测河中点 GeoJSON；不填则使用 outputs/corrected_rivers/校正河中点_全部断面.geojson",
    )
    parser.add_argument("--output-dir", default=None, help="输出目录；不填则使用 outputs/river_offset_diagnostics")
    parser.add_argument("--measured-crs", default="EPSG:4548", help="实测河中点 CRS，默认 EPSG:4548")
    parser.add_argument("--initial-search-m", type=float, default=500.0, help="查找最近原始河道的初始搜索半径")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def pick_text(row, fields: list[str]) -> str:
    for field in fields:
        if field in row.index:
            value = row.get(field)
            if value is None:
                continue
            text = str(value).strip()
            if text and text.lower() != "nan":
                return text
    return ""


def read_inputs(args: argparse.Namespace, config: dict) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Path]:
    river_path = Path(args.river_path or config.get("river_network_path") or "")
    if not river_path.exists():
        raise FileNotFoundError(f"原始河流水系不存在：{river_path}")

    measured_path = Path(args.measured_points or Path(config.get("output_dir") or "outputs") / "corrected_rivers" / "校正河中点_全部断面.geojson")
    if not measured_path.exists():
        raise FileNotFoundError(f"实测河中点文件不存在：{measured_path}")

    output_dir = Path(args.output_dir or Path(config.get("output_dir") or "outputs") / "river_offset_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    rivers = gpd.read_file(river_path)
    if rivers.empty:
        raise ValueError(f"原始河流水系为空：{river_path}")
    if rivers.crs is None:
        crs = config.get("river_network_crs")
        if not crs:
            raise ValueError("原始河流水系缺少 CRS，请在 config.yaml 中填写 river_network_crs。")
        rivers = rivers.set_crs(crs)

    points = gpd.read_file(measured_path)
    if points.empty:
        raise ValueError(f"实测河中点为空：{measured_path}")
    if points.crs is None:
        points = points.set_crs(args.measured_crs)

    rivers = rivers.to_crs(points.crs)
    return rivers, points, output_dir


def nearest_river_for_point(point, rivers: gpd.GeoDataFrame, initial_search_m: float):
    search = max(initial_search_m, 1.0)
    candidate_indices = []
    for _ in range(6):
        minx, miny, maxx, maxy = point.buffer(search).bounds
        candidate_indices = list(rivers.sindex.intersection((minx, miny, maxx, maxy)))
        if candidate_indices:
            break
        search *= 2.0
    if not candidate_indices:
        candidate_indices = list(rivers.index)

    candidates = rivers.loc[candidate_indices]
    distances = candidates.geometry.distance(point)
    nearest_index = distances.idxmin()
    nearest_row = rivers.loc[nearest_index]
    nearest_geom = nearest_row.geometry
    _point_on_measured, point_on_river = nearest_points(point, nearest_geom)
    return nearest_index, nearest_row, point_on_river


def compute_offsets(rivers: gpd.GeoDataFrame, points: gpd.GeoDataFrame, initial_search_m: float) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    rows = []
    vector_rows = []
    for point_index, row in points.iterrows():
        point = row.geometry
        if point is None or point.is_empty:
            continue
        nearest_index, nearest_row, point_on_river = nearest_river_for_point(point, rivers, initial_search_m)
        dx = float(point.x - point_on_river.x)
        dy = float(point.y - point_on_river.y)
        distance = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx))
        measured_river = str(row.get("river_folder", row.get("river_name", "")) or "")
        original_river = pick_text(nearest_row, RIVER_NAME_FIELDS)
        record = {
            "point_index": point_index,
            "measured_river_folder": measured_river,
            "measured_river_name": str(row.get("river_name", "")),
            "section_name": str(row.get("section_name", "")),
            "center_source": str(row.get("center_source", "")),
            "measured_x": float(point.x),
            "measured_y": float(point.y),
            "original_nearest_x": float(point_on_river.x),
            "original_nearest_y": float(point_on_river.y),
            "dx_to_move_original_m": dx,
            "dy_to_move_original_m": dy,
            "offset_distance_m": distance,
            "offset_angle_degree": angle,
            "nearest_original_index": nearest_index,
            "nearest_original_name": original_river,
        }
        rows.append(record)
        vector_record = record.copy()
        vector_record["geometry"] = LineString([(point_on_river.x, point_on_river.y), (point.x, point.y)])
        vector_rows.append(vector_record)

    offsets = pd.DataFrame(rows)
    vectors = gpd.GeoDataFrame(vector_rows, geometry="geometry", crs=points.crs)
    return offsets, vectors


def robust_summary(frame: pd.DataFrame, label: str) -> dict:
    if frame.empty:
        return {"group": label, "count": 0}
    med_dx = float(frame["dx_to_move_original_m"].median())
    med_dy = float(frame["dy_to_move_original_m"].median())
    residual = np.hypot(frame["dx_to_move_original_m"] - med_dx, frame["dy_to_move_original_m"] - med_dy)
    return {
        "group": label,
        "count": int(len(frame)),
        "dx_mean_m": float(frame["dx_to_move_original_m"].mean()),
        "dy_mean_m": float(frame["dy_to_move_original_m"].mean()),
        "dx_median_m": med_dx,
        "dy_median_m": med_dy,
        "dx_std_m": float(frame["dx_to_move_original_m"].std(ddof=0)),
        "dy_std_m": float(frame["dy_to_move_original_m"].std(ddof=0)),
        "offset_mean_m": float(frame["offset_distance_m"].mean()),
        "offset_median_m": float(frame["offset_distance_m"].median()),
        "offset_p90_m": float(frame["offset_distance_m"].quantile(0.9)),
        "residual_median_m": float(np.median(residual)),
        "residual_p90_m": float(np.quantile(residual, 0.9)),
        "recommended_global_dx_m": med_dx,
        "recommended_global_dy_m": med_dy,
    }


def recommendation(overall: dict, by_river: pd.DataFrame) -> str:
    residual_p90 = overall.get("residual_p90_m")
    if residual_p90 is None:
        return "无有效偏移样本。"
    stable_river_ratio = 0.0
    if not by_river.empty and "residual_p90_m" in by_river:
        stable_river_ratio = float((by_river["residual_p90_m"] <= 50).mean())
    if residual_p90 <= 50:
        return "整体偏移较集中，可以优先试用统一平移校正无断面河道。"
    if stable_river_ratio >= 0.7:
        return "整体偏移不完全统一，但多数河道内部偏移较集中，建议按河道或局部分区校正。"
    return "偏移离散较大，不建议直接用统一平移校正无断面河道；应使用分区校正或人工核查。"


def translate_original_rivers(rivers: gpd.GeoDataFrame, dx: float, dy: float) -> gpd.GeoDataFrame:
    moved = rivers.copy()
    moved["geometry"] = moved.geometry.apply(lambda geom: translate(geom, xoff=dx, yoff=dy) if geom is not None else geom)
    moved["correction_method"] = "global_median_translation_trial"
    moved["dx_m"] = dx
    moved["dy_m"] = dy
    return moved


def write_outputs(
    offsets: pd.DataFrame,
    vectors: gpd.GeoDataFrame,
    rivers_projected: gpd.GeoDataFrame,
    output_dir: Path,
) -> dict:
    if offsets.empty:
        raise ValueError("没有生成任何偏移记录。")
    overall = robust_summary(offsets, "all")
    by_measured_river = pd.DataFrame(
        [robust_summary(group, str(name)) for name, group in offsets.groupby("measured_river_folder", dropna=False)]
    ).sort_values(["count", "offset_median_m"], ascending=[False, False])
    by_original_river = pd.DataFrame(
        [robust_summary(group, str(name)) for name, group in offsets.groupby("nearest_original_name", dropna=False)]
    ).sort_values(["count", "offset_median_m"], ascending=[False, False])
    overall["recommendation"] = recommendation(overall, by_measured_river)

    excel_path = output_dir / "河道偏移诊断_断面河中点.xlsx"
    vector_path = output_dir / "河道偏移向量_断面河中点.geojson"
    translated_path = output_dir / "原始河道_整体平移校正_试算.geojson"
    params_path = output_dir / "偏移校正参数.yaml"

    vectors.to_file(vector_path, driver="GeoJSON", encoding="utf-8")
    translated = translate_original_rivers(
        rivers_projected,
        overall["recommended_global_dx_m"],
        overall["recommended_global_dy_m"],
    )
    translated.to_file(translated_path, driver="GeoJSON", encoding="utf-8")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame([overall]).to_excel(writer, sheet_name="overall_summary", index=False)
        by_measured_river.to_excel(writer, sheet_name="by_measured_river", index=False)
        by_original_river.to_excel(writer, sheet_name="by_original_river", index=False)
        offsets.to_excel(writer, sheet_name="point_offsets", index=False)

    params = {
        "recommended_global_dx_m": float(overall["recommended_global_dx_m"]),
        "recommended_global_dy_m": float(overall["recommended_global_dy_m"]),
        "residual_p90_m": float(overall["residual_p90_m"]),
        "offset_median_m": float(overall["offset_median_m"]),
        "recommendation": overall["recommendation"],
        "note": "dx/dy 表示把原始河道移动到实测河中点附近的平移量；试算文件不应直接覆盖原始数据。",
    }
    with params_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(params, file, allow_unicode=True, sort_keys=False)

    return {
        "excel_path": excel_path,
        "vector_path": vector_path,
        "translated_path": translated_path,
        "params_path": params_path,
        "overall": overall,
        "by_measured_river": by_measured_river,
    }


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))
    rivers, points, output_dir = read_inputs(args, config)
    offsets, vectors = compute_offsets(rivers, points, args.initial_search_m)
    outputs = write_outputs(offsets, vectors, rivers, output_dir)
    overall = outputs["overall"]
    print(f"偏移样本数：{overall['count']}")
    print(f"整体中位偏移 dx={overall['recommended_global_dx_m']:.3f} m, dy={overall['recommended_global_dy_m']:.3f} m")
    print(f"偏移距离中位数：{overall['offset_median_m']:.3f} m；残差 P90：{overall['residual_p90_m']:.3f} m")
    print(f"建议：{overall['recommendation']}")
    print(f"诊断表：{outputs['excel_path']}")
    print(f"偏移向量：{outputs['vector_path']}")
    print(f"整体平移试算河道：{outputs['translated_path']}")
    print(f"参数文件：{outputs['params_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
