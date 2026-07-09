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
import rasterio
import yaml

from src.flood_analysis import run_analysis
from src.logger_utils import WorkflowLogger
from src.raster_utils import section_value_stats


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "web_static"
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


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
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def coerce_value(key: str, value):
    if value == "":
        return None
    if key in {"threshold", "time_interval_hours"}:
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
