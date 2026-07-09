from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window, from_bounds, transform as window_transform
from rasterstats import zonal_stats
from shapely.geometry.base import BaseGeometry


def open_raster_crs(path: Path, fallback_crs: str | None = None) -> CRS | None:
    with rasterio.open(path) as src:
        if src.crs:
            return src.crs
    return CRS.from_user_input(fallback_crs) if fallback_crs else None


def raster_bounds(path: Path) -> tuple[float, float, float, float]:
    with rasterio.open(path) as src:
        b = src.bounds
        return b.left, b.bottom, b.right, b.top


def crop_raster_to_temp(
    raster_path: Path,
    bounds: tuple[float, float, float, float],
    output_dir: Path,
    raster_crs: CRS | str | None,
    padding_pixels: int = 2,
) -> Path:
    """Crop a large raster to the analysis bounds and write a small GeoTIFF."""

    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1((str(raster_path.resolve()) + repr(bounds)).encode("utf-8")).hexdigest()[:12]
    out_path = output_dir / f"{raster_path.stem}_{digest}.tif"
    if out_path.exists():
        return out_path

    with rasterio.open(raster_path) as src:
        window = from_bounds(*bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()
        window = Window(
            max(0, window.col_off - padding_pixels),
            max(0, window.row_off - padding_pixels),
            window.width + padding_pixels * 2,
            window.height + padding_pixels * 2,
        )
        full = Window(0, 0, src.width, src.height)
        window = window.intersection(full)
        if window.width <= 0 or window.height <= 0:
            raise ValueError(f"栅格与分析范围没有重叠：{raster_path}")

        data = src.read(1, window=window)
        profile: dict[str, Any] = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            transform=window_transform(window, src.transform),
            compress="lzw",
        )
        if raster_crs and not profile.get("crs"):
            profile["crs"] = CRS.from_user_input(raster_crs)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
    return out_path


def _clean_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def zonal_max_mean(
    geometries: list[BaseGeometry],
    raster_path: Path,
    all_touched: bool = False,
    fill_none_with_zero: bool = True,
) -> list[dict[str, float | None]]:
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
    stats = zonal_stats(
        geometries,
        str(raster_path),
        stats=["max", "mean"],
        nodata=nodata,
        all_touched=all_touched,
        geojson_out=False,
    )
    rows: list[dict[str, float | None]] = []
    for row in stats:
        max_v = _clean_value(row.get("max"))
        mean_v = _clean_value(row.get("mean"))
        if fill_none_with_zero:
            max_v = 0.0 if max_v is None else max_v
            mean_v = 0.0 if mean_v is None else mean_v
        rows.append({"max": max_v, "mean": mean_v})
    return rows


def sample_points(
    points: list[BaseGeometry],
    raster_path: Path,
    fill_none_with_zero: bool = True,
) -> list[dict[str, float | None]]:
    coords = [(geom.x, geom.y) for geom in points]
    rows: list[dict[str, float | None]] = []
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        for sample in src.sample(coords):
            value = _clean_value(sample[0])
            if nodata is not None and value == nodata:
                value = None
            if fill_none_with_zero:
                value = 0.0 if value is None else value
            rows.append({"max": value, "mean": value})
    return rows


def section_value_stats(
    geometry: BaseGeometry,
    raster_path: Path,
    method: str,
    buffer_m: float = 5.0,
    all_touched: bool = True,
) -> dict[str, float | None]:
    if geometry.is_empty:
        return {"max": None, "mean": None, "median": None}

    geom_type = geometry.geom_type.lower()
    if "point" in geom_type:
        with rasterio.open(raster_path) as src:
            value = _clean_value(next(src.sample([(geometry.x, geometry.y)]))[0])
            if src.nodata is not None and value == src.nodata:
                value = None
        return {"max": value, "mean": value, "median": value}

    stat_geom = geometry
    if "line" in geom_type:
        stat_geom = geometry.buffer(buffer_m)

    with rasterio.open(raster_path) as src:
        nodata = src.nodata
    rows = zonal_stats(
        [stat_geom],
        str(raster_path),
        stats=["max", "mean", "median"],
        nodata=nodata,
        all_touched=all_touched,
        geojson_out=False,
    )
    row = rows[0] if rows else {}
    return {key: _clean_value(row.get(key)) for key in ("max", "mean", "median")}
