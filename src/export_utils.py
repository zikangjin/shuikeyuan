from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd


def safe_filename(value: str) -> str:
    text = re.sub(r'[\\/:*?"<>|]+', "_", str(value).strip())
    return text or "未命名"


def compact_list(values, limit: int = 200) -> str:
    items = [str(v) for v in values]
    if len(items) <= limit:
        return "、".join(items)
    return "、".join(items[:limit]) + f" ... 共 {len(items)} 个，详见明细 sheet"


def write_outputs(
    summary: dict,
    building_details: pd.DataFrame,
    flooded_details: pd.DataFrame,
    output_dir: Path,
    logger,
    section_distances: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    village = safe_filename(summary.get("村庄", "村庄"))
    scenario = safe_filename(summary.get("情景", "情景"))
    base = f"淹没分析结果_{village}_{scenario}"
    xlsx_path = output_dir / f"{base}.xlsx"
    csv_path = output_dir / f"{base}.csv"

    summary_df = pd.DataFrame([summary])
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            building_details.to_excel(writer, sheet_name="buildings", index=False)
            flooded_details.to_excel(writer, sheet_name="first_flooded_buildings", index=False)
            if section_distances is not None and not section_distances.empty:
                section_distances.to_excel(writer, sheet_name="section_distances", index=False)
    except PermissionError:
        stamped = datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx_path = output_dir / f"{base}_{stamped}.xlsx"
        logger.warning(f"Excel 文件可能被占用，改写入新文件：{xlsx_path}")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            building_details.to_excel(writer, sheet_name="buildings", index=False)
            flooded_details.to_excel(writer, sheet_name="first_flooded_buildings", index=False)
            if section_distances is not None and not section_distances.empty:
                section_distances.to_excel(writer, sheet_name="section_distances", index=False)

    try:
        summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        stamped = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"{base}_{stamped}.csv"
        logger.warning(f"CSV 文件可能被占用，改写入新文件：{csv_path}")
        summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return xlsx_path, csv_path
