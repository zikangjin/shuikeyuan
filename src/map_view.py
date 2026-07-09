from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import GeometryCollection, MultiLineString, MultiPolygon


class FloodMapView(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=6)
        self.payload: dict | None = None
        self.frame_index = 0
        self.playing = False
        self.extent: tuple[float, float, float, float] | None = None
        self.home_extent: tuple[float, float, float, float] | None = None
        self.drag_start: tuple[int, int] | None = None
        self.drag_extent: tuple[float, float, float, float] | None = None
        self.show_shallow = tk.BooleanVar(value=False)
        self.show_rivers = tk.BooleanVar(value=True)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, bg="#f7f9fb", highlightthickness=1, highlightbackground="#c7ced8", cursor="fleur")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        controls = ttk.Frame(self)
        controls.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(controls, text="播放/暂停", command=self.toggle_play).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="上一时刻", command=self.previous_frame).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="下一时刻", command=self.next_frame).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(controls, text="放大", command=lambda: self.zoom(0.75)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="缩小", command=lambda: self.zoom(1.35)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="复位", command=self.reset_view).pack(side=tk.LEFT, padx=(0, 12))
        self.shallow_button = ttk.Button(controls, text="显示浅水", command=self.toggle_shallow)
        self.shallow_button.pack(side=tk.LEFT, padx=(0, 12))
        self.river_button = ttk.Button(controls, text="隐藏河流", command=self.toggle_rivers)
        self.river_button.pack(side=tk.LEFT, padx=(0, 12))
        self.info_var = tk.StringVar(value="等待分析结果")
        ttk.Label(controls, textvariable=self.info_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.canvas.bind("<Configure>", lambda _event: self.render())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(0.82, event.x, event.y))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(1.22, event.x, event.y))
        self.canvas.bind("<ButtonPress-1>", self._start_pan)
        self.canvas.bind("<B1-Motion>", self._pan)
        self.canvas.bind("<ButtonRelease-1>", self._end_pan)
        self.show_message("开始分析后，这里会显示村庄建筑、附近断面和淹没过程动画。")

    def show_message(self, message: str) -> None:
        self.payload = None
        self.canvas.delete("all")
        self.info_var.set(message)
        self.canvas.create_text(
            max(self.canvas.winfo_width() // 2, 300),
            max(self.canvas.winfo_height() // 2, 180),
            text=message,
            fill="#566274",
            font=("Microsoft YaHei UI", 12),
            width=520,
            justify=tk.CENTER,
        )

    def load_payload(self, payload: dict) -> None:
        self.payload = payload or {}
        self.frame_index = 0
        self.playing = True
        self.home_extent = self._compute_extent()
        self.extent = self.home_extent
        self.render()
        self.after(700, self._tick)

    def toggle_play(self) -> None:
        if not self.payload:
            return
        self.playing = not self.playing
        if self.playing:
            self.after(100, self._tick)

    def toggle_shallow(self) -> None:
        self.show_shallow.set(not self.show_shallow.get())
        self.shallow_button.configure(text="隐藏浅水" if self.show_shallow.get() else "显示浅水")
        self.render()

    def toggle_rivers(self) -> None:
        self.show_rivers.set(not self.show_rivers.get())
        self.river_button.configure(text="隐藏河流" if self.show_rivers.get() else "显示河流")
        self.render()

    def previous_frame(self) -> None:
        if not self.payload:
            return
        frames = self.payload.get("frames") or []
        if not frames:
            return
        self.playing = False
        self.frame_index = (self.frame_index - 1) % len(frames)
        self.render()

    def next_frame(self) -> None:
        if not self.payload:
            return
        frames = self.payload.get("frames") or []
        if not frames:
            return
        self.playing = False
        self.frame_index = (self.frame_index + 1) % len(frames)
        self.render()

    def _tick(self) -> None:
        if not self.payload or not self.playing:
            return
        frames = self.payload.get("frames") or []
        if frames:
            self.frame_index = (self.frame_index + 1) % len(frames)
            self.render()
        self.after(850, self._tick)

    def _compute_extent(self) -> tuple[float, float, float, float] | None:
        if not self.payload:
            return None
        bounds = []
        buildings = self.payload.get("buildings")
        sections = self.payload.get("sections")
        rivers = self.payload.get("rivers")
        if buildings is not None and not buildings.empty:
            bounds.append(tuple(buildings.total_bounds))
        if sections is not None and not sections.empty:
            bounds.append(tuple(sections.total_bounds))
        if rivers is not None and not rivers.empty:
            bounds.append(tuple(rivers.total_bounds))
        for frame in self.payload.get("frames") or []:
            path = frame.get("raster_path")
            if not path:
                continue
            try:
                with rasterio.open(path) as src:
                    b = src.bounds
                    bounds.append((b.left, b.bottom, b.right, b.top))
            except Exception:
                continue
        if not bounds:
            return None
        minx = min(b[0] for b in bounds)
        miny = min(b[1] for b in bounds)
        maxx = max(b[2] for b in bounds)
        maxy = max(b[3] for b in bounds)
        dx = max(maxx - minx, 1.0)
        dy = max(maxy - miny, 1.0)
        pad = max(dx, dy) * 0.06
        return minx - pad, miny - pad, maxx + pad, maxy + pad

    def _view_transform(self) -> tuple[float, float, float, float, float, float, float]:
        assert self.extent is not None
        minx, miny, maxx, maxy = self.extent
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        margin = 18
        sx = max(width - margin * 2, 1) / max(maxx - minx, 1e-9)
        sy = max(height - margin * 2, 1) / max(maxy - miny, 1e-9)
        scale = min(sx, sy)
        draw_w = (maxx - minx) * scale
        draw_h = (maxy - miny) * scale
        ox = (width - draw_w) / 2
        oy = (height - draw_h) / 2
        return minx, miny, maxx, maxy, scale, ox, oy

    def _xy(self, x: float, y: float) -> tuple[float, float]:
        minx, _miny, _maxx, maxy, scale, ox, oy = self._view_transform()
        return ox + (x - minx) * scale, oy + (maxy - y) * scale

    def _world_xy(self, sx: float, sy: float) -> tuple[float, float]:
        minx, _miny, _maxx, maxy, scale, ox, oy = self._view_transform()
        return minx + (sx - ox) / scale, maxy - (sy - oy) / scale

    def _on_mousewheel(self, event) -> None:
        if event.delta > 0:
            self._zoom_at(0.82, event.x, event.y)
        elif event.delta < 0:
            self._zoom_at(1.22, event.x, event.y)

    def zoom(self, factor: float) -> None:
        self._zoom_at(factor, self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)

    def _zoom_at(self, factor: float, screen_x: float, screen_y: float) -> None:
        if not self.payload or self.extent is None:
            return
        minx, miny, maxx, maxy = self.extent
        world_x, world_y = self._world_xy(screen_x, screen_y)
        width = max(maxx - minx, 1e-9)
        height = max(maxy - miny, 1e-9)
        new_width = width * factor
        new_height = height * factor
        if self.home_extent is not None:
            home_width = max(self.home_extent[2] - self.home_extent[0], 1e-9)
            home_height = max(self.home_extent[3] - self.home_extent[1], 1e-9)
            min_width = home_width / 80
            min_height = home_height / 80
            max_width = home_width * 8
            max_height = home_height * 8
            new_width = min(max(new_width, min_width), max_width)
            new_height = min(max(new_height, min_height), max_height)
        rx = (world_x - minx) / width
        ry = (world_y - miny) / height
        self.extent = (
            world_x - rx * new_width,
            world_y - ry * new_height,
            world_x + (1 - rx) * new_width,
            world_y + (1 - ry) * new_height,
        )
        self.render()

    def reset_view(self) -> None:
        if self.home_extent is None:
            return
        self.extent = self.home_extent
        self.render()

    def _start_pan(self, event) -> None:
        if not self.payload or self.extent is None:
            return
        self.drag_start = (event.x, event.y)
        self.drag_extent = self.extent

    def _pan(self, event) -> None:
        if self.drag_start is None or self.drag_extent is None or self.extent is None:
            return
        _minx, _miny, _maxx, _maxy, scale, _ox, _oy = self._view_transform()
        dx = (event.x - self.drag_start[0]) / scale
        dy = (event.y - self.drag_start[1]) / scale
        start_minx, start_miny, start_maxx, start_maxy = self.drag_extent
        self.extent = (start_minx - dx, start_miny + dy, start_maxx - dx, start_maxy + dy)
        self.render()

    def _end_pan(self, _event) -> None:
        self.drag_start = None
        self.drag_extent = None

    def render(self) -> None:
        self.canvas.delete("all")
        if not self.payload:
            return
        if self.extent is None:
            self.show_message("没有可显示的地图数据。")
            return
        frames = self.payload.get("frames") or []
        frame = frames[self.frame_index] if frames else None
        if frame:
            self._draw_flood_frame(frame)
        self._draw_rivers()
        self._draw_corrected_rivers()
        self._draw_measured_river_centers()
        self._draw_buildings()
        self._draw_reference_buildings()
        self._draw_sections(frame)
        self._draw_legend()
        self._update_info(frame)

    def _draw_flood_frame(self, frame: dict) -> None:
        path = frame.get("raster_path")
        if not path or not Path(path).exists():
            return
        threshold = float(self.payload.get("threshold") or 0)
        shallow_min = max(threshold * 0.25, 0.02) if threshold > 0 else 0.02
        try:
            with rasterio.open(path) as src:
                data = src.read(1, masked=True)
                transform = src.transform
        except Exception:
            return
        if data.size == 0:
            return
        height, width = data.shape
        step = max(1, int(math.ceil(max(height, width) / 120)))
        for row in range(0, height, step):
            for col in range(0, width, step):
                block = data[row : min(row + step, height), col : min(col + step, width)]
                if np.ma.is_masked(block):
                    if block.mask.all():
                        continue
                    value = float(block.max())
                else:
                    value = float(np.nanmax(block))
                if not math.isfinite(value) or value <= 0:
                    continue
                if value >= threshold:
                    color = "#2f80ed"
                elif self.show_shallow.get() and value >= shallow_min:
                    color = "#a9d6ff"
                else:
                    continue
                x0, y0 = transform * (col, row)
                x1, y1 = transform * (min(col + step, width), min(row + step, height))
                c0 = self._xy(x0, y0)
                c1 = self._xy(x1, y1)
                self.canvas.create_rectangle(c0[0], c0[1], c1[0], c1[1], fill=color, outline="", tags="flood")

    def _draw_buildings(self) -> None:
        buildings = self.payload.get("buildings")
        if buildings is None or buildings.empty:
            return
        for geom in buildings.geometry:
            for coords in self._polygon_rings(geom):
                points = []
                for x, y in coords:
                    points.extend(self._xy(x, y))
                if len(points) >= 6:
                    self.canvas.create_polygon(points, fill="", outline="#202832", width=1)

    def _draw_rivers(self) -> None:
        if not self.show_rivers.get():
            return
        rivers = self.payload.get("rivers") if self.payload else None
        if rivers is None or rivers.empty:
            return
        for _, row in rivers.iterrows():
            for coords in self._line_coords(row.geometry):
                points = []
                for x, y in coords:
                    points.extend(self._xy(x, y))
                if len(points) >= 4:
                    self.canvas.create_line(points, fill="#0077b6", width=2, smooth=True)
            self._draw_river_label(row)

    def _draw_corrected_rivers(self) -> None:
        if not self.show_rivers.get():
            return
        rivers = self.payload.get("corrected_rivers") if self.payload else None
        if rivers is None or rivers.empty:
            return
        for _, row in rivers.iterrows():
            for coords in self._line_coords(row.geometry):
                points = []
                for x, y in coords:
                    points.extend(self._xy(x, y))
                if len(points) >= 4:
                    self.canvas.create_line(points, fill="#00a676", width=3, smooth=True)
            anchor = self._label_point(row.geometry)
            if anchor is not None:
                x, y = self._xy(anchor.x, anchor.y)
                self._draw_inline_label(x, y, "校正河道", fill="#006d4f", font=("Microsoft YaHei UI", 8, "bold"), anchor="center")

    def _draw_measured_river_centers(self) -> None:
        sections = self.payload.get("sections") if self.payload else None
        if sections is None or sections.empty:
            return
        groups: dict[str, list[tuple[int, float, float, str]]] = {}
        nearest = str(self.payload.get("nearest_section_id") or "")
        for _, row in sections.iterrows():
            x = row.get("section_river_center_map_x")
            y = row.get("section_river_center_map_y")
            if not isinstance(x, (int, float, np.number)) or not isinstance(y, (int, float, np.number)):
                continue
            if not math.isfinite(float(x)) or not math.isfinite(float(y)):
                continue
            section_id = str(row.get("section_id", ""))
            group = str(row.get("river_folder", "") or self._short_section_id(section_id, ""))
            groups.setdefault(group, []).append((self._section_order(section_id), float(x), float(y), section_id))

        for points in groups.values():
            points.sort(key=lambda item: item[0])
            line_points = []
            for _order, x, y, _section_id in points:
                line_points.extend(self._xy(x, y))
            if len(line_points) >= 4:
                self.canvas.create_line(line_points, fill="#00a6a6", width=2, dash=(5, 4), smooth=True)
            for _order, x, y, section_id in points:
                sx, sy = self._xy(x, y)
                radius = 4 if section_id == nearest else 3
                self.canvas.create_oval(
                    sx - radius,
                    sy - radius,
                    sx + radius,
                    sy + radius,
                    fill="#00c2c7",
                    outline="#006d77",
                    width=1,
                )
                if section_id == nearest:
                    self._draw_inline_label(sx + 8, sy - 8, "实测河中", fill="#006d77", font=("Microsoft YaHei UI", 8, "bold"), anchor="w")

    def _draw_reference_buildings(self) -> None:
        buildings = self.payload.get("reference_buildings")
        if buildings is None or buildings.empty:
            return
        for geom in buildings.geometry:
            for coords in self._polygon_rings(geom):
                points = []
                for x, y in coords:
                    points.extend(self._xy(x, y))
                if len(points) >= 6:
                    self.canvas.create_polygon(points, fill="", outline="#8e44ad", width=3)

    def _draw_sections(self, frame: dict | None = None) -> None:
        sections = self.payload.get("sections")
        if sections is None or sections.empty:
            return
        nearest = str(self.payload.get("nearest_section_id") or "")
        value_label = self.payload.get("value_label") or "水深"
        raster_path = frame.get("raster_path") if frame else None
        buffer_m = float(self.payload.get("section_buffer_m") or 5)
        for _, row in sections.iterrows():
            section_id = str(row.get("section_id", ""))
            color = "#c0392b" if section_id == nearest else "#f39c12"
            width = 3 if section_id == nearest else 2
            for coords in self._line_coords(row.geometry):
                points = []
                for x, y in coords:
                    points.extend(self._xy(x, y))
                if len(points) >= 4:
                    self.canvas.create_line(points, fill=color, width=width, smooth=True)
            self._draw_section_label(row, raster_path, buffer_m, value_label, color)

    def _draw_section_label(self, row, raster_path: str | None, buffer_m: float, value_label: str, color: str) -> None:
        geom = row.geometry
        if geom is None or geom.is_empty:
            return
        anchor = self._label_point(geom)
        if anchor is None:
            return
        x, y = self._xy(anchor.x, anchor.y)
        section_id = self._field_text(row, ["section_id", "断面号", "断面编号", "编号", "id", "ID"]) or "未命名断面"
        river_name = self._field_text(row, self._river_name_fields()) or ""
        section_value = self._section_raster_value(geom, raster_path, buffer_m)
        value_text = "-" if section_value is None else f"{section_value:.2f}m"
        short_id = self._short_section_id(section_id, river_name)
        parts = [short_id]
        if river_name and river_name not in short_id:
            parts.append(river_name)
        trimmed_length = row.get("trimmed_length_m", None)
        if isinstance(trimmed_length, (int, float, np.number)) and math.isfinite(float(trimmed_length)):
            parts.append(f"长{float(trimmed_length):.0f}m")
        original_depth = row.get("section_original_depth_m", None)
        if isinstance(original_depth, (int, float, np.number)) and math.isfinite(float(original_depth)):
            parts.append(f"原深{float(original_depth):.2f}m")
        parts.append(f"{value_label}:{value_text}")
        label = " | ".join(part for part in parts if part)
        fill = "#8b1e12" if color == "#c0392b" else "#6b4300"
        self._draw_inline_label(
            x,
            y,
            text=label,
            fill=fill,
            font=("Microsoft YaHei UI", 8),
            anchor="center",
        )

    def _draw_river_label(self, row) -> None:
        river_name = self._field_text(row, self._river_name_fields())
        if not river_name:
            return
        anchor = self._label_point(row.geometry)
        if anchor is None:
            return
        x, y = self._xy(anchor.x, anchor.y)
        self._draw_inline_label(
            x,
            y,
            text=river_name,
            fill="#006494",
            font=("Microsoft YaHei UI", 8, "bold"),
            anchor="center",
        )

    def _draw_inline_label(self, x: float, y: float, text: str, fill: str, font, anchor: str = "center") -> None:
        if not text:
            return
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            self.canvas.create_text(x + dx, y + dy, text=text, anchor=anchor, fill="#ffffff", font=font)
        self.canvas.create_text(x, y, text=text, anchor=anchor, fill=fill, font=font)

    def _short_section_id(self, section_id: str, river_name: str) -> str:
        section_id = str(section_id or "").strip()
        river_name = str(river_name or "").strip()
        if river_name and section_id.startswith(f"{river_name}_"):
            return section_id[len(river_name) + 1 :]
        if river_name and section_id.startswith(f"{river_name}-"):
            return section_id[len(river_name) + 1 :]
        return section_id or "未命名断面"

    def _section_order(self, section_id: str) -> int:
        digits = ""
        for char in reversed(str(section_id or "")):
            if char.isdigit():
                digits = char + digits
            elif digits:
                break
        return int(digits) if digits else 0

    def _section_raster_value(self, geom, raster_path: str | None, buffer_m: float) -> float | None:
        if not raster_path or not Path(raster_path).exists() or geom is None or geom.is_empty:
            return None
        try:
            with rasterio.open(raster_path) as src:
                if geom.geom_type.lower() == "point":
                    value = next(src.sample([(geom.x, geom.y)], masked=True))[0]
                    if np.ma.is_masked(value) or not math.isfinite(float(value)):
                        return None
                    return float(value)
                sample_geom = geom.buffer(buffer_m) if buffer_m > 0 else geom
                data, _transform = mask(src, [sample_geom.__geo_interface__], crop=True, filled=False)
        except Exception:
            return None
        values = np.ma.masked_invalid(data[0])
        if values.count() == 0:
            return None
        max_value = float(values.max())
        return max_value if math.isfinite(max_value) else None

    def _label_point(self, geom):
        geom_type = geom.geom_type.lower()
        try:
            if "line" in geom_type and geom.length > 0:
                return geom.interpolate(0.5, normalized=True)
            if geom_type == "point":
                return geom
            return geom.representative_point()
        except Exception:
            return None

    def _field_text(self, row, candidates: list[str]) -> str:
        for field in candidates:
            if field in row.index:
                value = row.get(field)
                if value is None:
                    continue
                text = str(value).strip()
                if text and text.lower() != "nan":
                    return text
        return ""

    def _river_name_fields(self) -> list[str]:
        return [
            "river_folder",
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

    def _draw_legend(self) -> None:
        x, y = 16, 16
        items = [
            ("#0077b6", "原河流水系"),
            ("#00a676", "校正河道"),
            ("#00a6a6", "实测河中"),
            ("#202832", "建筑物"),
            ("#8e44ad", "距离参考建筑"),
            ("#f39c12", "附近断面"),
            ("#c0392b", "最近断面"),
            ("#2f80ed", "达到阈值"),
        ]
        if not (self.payload and self.payload.get("rivers") is not None and self.show_rivers.get()):
            items = [item for item in items if item[1] != "原河流水系"]
        if not (self.payload and self.payload.get("corrected_rivers") is not None and self.show_rivers.get()):
            items = [item for item in items if item[1] != "校正河道"]
        if self.show_shallow.get():
            items.append(("#a9d6ff", "浅水未达阈值"))
        for color, text in items:
            self.canvas.create_rectangle(x, y, x + 16, y + 10, fill=color, outline="")
            self.canvas.create_text(x + 22, y + 5, text=text, anchor="w", fill="#2f3640", font=("Microsoft YaHei UI", 9))
            y += 18

    def _update_info(self, frame: dict | None) -> None:
        if not frame:
            self.info_var.set("没有动画帧")
            return
        total = len(self.payload.get("frames") or [])
        value_label = self.payload.get("value_label") or "水深"
        first_flood = self.payload.get("first_flood_time") or "未受淹"
        nearest = self.payload.get("nearest_section_id") or ""
        max_value = frame.get("max_value")
        max_text = f"{max_value:.3f}" if isinstance(max_value, (int, float)) and math.isfinite(max_value) else "-"
        area_ratio = frame.get("threshold_area_ratio_pct")
        area_text = f" | 达阈值面积 {area_ratio:.2f}%" if isinstance(area_ratio, (int, float)) and math.isfinite(area_ratio) else ""
        mode_parts = ["含浅水" if self.show_shallow.get() else "只显示达阈值"]
        if self.payload.get("rivers") is not None:
            mode_parts.append("显示河流" if self.show_rivers.get() else "隐藏河流")
        mode = " / ".join(mode_parts)
        nearest_text = nearest or "无有效附近断面"
        self.info_var.set(
            f"{frame.get('label')} ({self.frame_index + 1}/{total}) | "
            f"受淹建筑物 {frame.get('flooded_count')} | 最大{value_label} {max_text}{area_text} | "
            f"首次受淹 {first_flood} | 最近断面 {nearest_text} | {mode}"
        )

    def _polygon_rings(self, geom):
        if geom is None or geom.is_empty:
            return
        if isinstance(geom, MultiPolygon):
            for part in geom.geoms:
                yield list(part.exterior.coords)
        elif isinstance(geom, GeometryCollection):
            for part in geom.geoms:
                yield from self._polygon_rings(part)
        elif geom.geom_type == "Polygon":
            yield list(geom.exterior.coords)

    def _line_coords(self, geom):
        if geom is None or geom.is_empty:
            return
        if isinstance(geom, MultiLineString):
            for part in geom.geoms:
                yield list(part.coords)
        elif isinstance(geom, GeometryCollection):
            for part in geom.geoms:
                yield from self._line_coords(part)
        elif geom.geom_type == "LineString":
            yield list(geom.coords)
        elif geom.geom_type == "Point":
            x, y = geom.x, geom.y
            delta = max((self.extent[2] - self.extent[0]) if self.extent else 1, 1) * 0.002
            yield [(x - delta, y), (x + delta, y)]
