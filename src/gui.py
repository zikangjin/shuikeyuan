from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import yaml

from .flood_analysis import run_analysis
from .logger_utils import WorkflowLogger
from .map_view import FloodMapView


SECTION_REFERENCE_OPTIONS = ["首次受淹建筑物", "最深受淹建筑物", "自动：优先村庄边界", "仅村庄边界", "建筑物整体"]
SECTION_REFERENCE_TO_VALUE = {
    "首次受淹建筑物": "first_flooded_buildings",
    "最深受淹建筑物": "first_flooded_max_building",
    "自动：优先村庄边界": "village_boundary_or_buildings",
    "仅村庄边界": "village_boundary",
    "建筑物整体": "buildings_geometry",
}
SECTION_REFERENCE_TO_LABEL = {value: label for label, value in SECTION_REFERENCE_TO_VALUE.items()}


class FloodWorkflowApp(ttk.Frame):
    def __init__(self, master: tk.Tk, config: dict | None = None):
        super().__init__(master, padding=10)
        self.master = master
        self.config_data = config or {}
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.entries: dict[str, tk.StringVar] = {}
        self.path_label_vars: dict[str, tk.StringVar] = {}
        self.value_type = tk.StringVar(value=str(self.config_data.get("value_type") or "水深"))
        section_reference_value = str(self.config_data.get("section_reference_mode") or "first_flooded_buildings")
        self.section_reference_mode = tk.StringVar(value=SECTION_REFERENCE_TO_LABEL.get(section_reference_value, section_reference_value))
        self.advanced_visible = tk.BooleanVar(value=False)
        self.running = False
        self.grid(sticky="nsew")
        master.title("村庄淹没与最近断面分析工具")
        master.geometry("1020x800")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self._build_ui()
        self.after(150, self._poll_logs)

    def _add_path_row(self, parent, row: int, label: str, key: str, mode: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
        var = self._var(key)
        display_var = tk.StringVar(value=self._path_display_text(var.get()))
        self.path_label_vars[key] = display_var
        ttk.Label(parent, textvariable=display_var, width=34, foreground="#4d5968").grid(row=row, column=1, sticky="w", padx=4, pady=4)
        if mode == "multi_file_or_dir":
            ttk.Button(parent, text="选择文件(可多选)", command=lambda: self._browse(key, "multi_files")).grid(row=row, column=2, padx=4, pady=4)
            ttk.Button(parent, text="添加文件夹", command=lambda: self._browse(key, "append_dir")).grid(row=row, column=3, padx=4, pady=4)
            return
        text = "选择文件夹" if mode == "dir" else "选择文件"
        ttk.Button(parent, text=text, command=lambda: self._browse(key, mode)).grid(row=row, column=2, padx=4, pady=4)

    def _add_entry_row(self, parent, row: int, label: str, key: str, width: int = 18) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(parent, textvariable=self._var(key), width=width).grid(row=row, column=1, sticky="ew", padx=4, pady=4)

    def _var(self, key: str) -> tk.StringVar:
        if key not in self.entries:
            value = self.config_data.get(key)
            if value is None and key == "section_path":
                value = self.config_data.get("section_paths")
            if isinstance(value, (list, tuple)):
                value = ";".join(str(item) for item in value)
            self.entries[key] = tk.StringVar(value="" if value is None else str(value))
        return self.entries[key]

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        data_frame = ttk.LabelFrame(self, text="数据选择区", padding=8)
        data_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._add_path_row(data_frame, 0, "村庄建筑物数据", "building_path", "file")
        self._add_path_row(data_frame, 1, "模拟淹没数据文件夹", "flood_folder", "dir")
        self._add_path_row(data_frame, 2, "河道断面数据（可多选）", "section_path", "multi_file_or_dir")
        self._add_path_row(data_frame, 3, "村庄边界数据（用于确定建筑）", "auxiliary_path", "file_or_dir")
        self._add_path_row(data_frame, 4, "输出文件夹（可空）", "output_dir", "dir")

        param_frame = ttk.LabelFrame(self, text="参数输入区", padding=8)
        param_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        for i in range(4):
            param_frame.columnconfigure(i * 2 + 1, weight=1)
        self._add_entry_row(param_frame, 0, "村名", "village_name")
        ttk.Label(param_frame, text="受淹阈值(m)").grid(row=0, column=2, sticky="w", padx=4)
        ttk.Entry(param_frame, textvariable=self._var("threshold"), width=12).grid(row=0, column=3, sticky="ew", padx=4)
        ttk.Label(param_frame, text="前序时间间隔(h)").grid(row=0, column=4, sticky="w", padx=4)
        ttk.Entry(param_frame, textvariable=self._var("time_interval_hours"), width=12).grid(row=0, column=5, sticky="ew", padx=4)

        ttk.Label(param_frame, text="淹没数据类型").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(param_frame, textvariable=self.value_type, values=["水深", "水位"], state="readonly", width=12).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(param_frame, text="情景名称").grid(row=1, column=2, sticky="w", padx=4)
        ttk.Entry(param_frame, textvariable=self._var("scenario_name"), width=18).grid(row=1, column=3, sticky="ew", padx=4)
        ttk.Label(
            param_frame,
            text="如果建筑物没有村名字段，请选择村庄边界数据，用它裁剪出该村建筑物。",
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=4, pady=4)
        self.advanced_button = ttk.Button(param_frame, text="显示高级参数", command=self._toggle_advanced)
        self.advanced_button.grid(row=2, column=4, columnspan=2, sticky="e", padx=4, pady=4)

        self.advanced_frame = ttk.LabelFrame(self, text="高级参数（一般不用改）", padding=8)
        advanced = self.advanced_frame
        for i in range(4):
            advanced.columnconfigure(i * 2 + 1, weight=1)
        ttk.Label(advanced, text="栅格 CRS（可空自动）").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(advanced, textvariable=self._var("raster_crs"), width=16).grid(row=0, column=1, sticky="ew", padx=4, pady=3)
        ttk.Label(advanced, text="最近断面参考").grid(row=0, column=2, sticky="w", padx=4, pady=3)
        ttk.Combobox(
            advanced,
            textvariable=self.section_reference_mode,
            values=SECTION_REFERENCE_OPTIONS,
            state="readonly",
            width=20,
        ).grid(row=0, column=3, sticky="ew", padx=4, pady=3)
        ttk.Label(
            advanced,
            text="规则：固定当前村庄，遍历所有断面，按所选参考对象到断面的最短距离排序。",
        ).grid(row=0, column=4, columnspan=4, sticky="w", padx=4, pady=3)
        labels = [
            ("建筑物村名字段", "building_village_field"),
            ("建筑物编号字段", "building_id_field"),
            ("建筑物预筛字段", "building_prefilter_field"),
            ("建筑物预筛值", "building_prefilter_value"),
            ("辅助村名字段", "auxiliary_village_field"),
            ("辅助编码", "auxiliary_encoding"),
            ("断面 CRS（可空）", "section_crs"),
            ("目标投影 CRS（可空）", "target_projected_crs"),
            ("断面编号字段", "section_id_field"),
            ("断面名称字段", "section_name_field"),
            ("断面 buffer(m)", "section_buffer_m"),
            ("断面最大距离(m)", "max_section_distance_m"),
            ("断面裁剪模式", "section_trim_mode"),
            ("断面裁剪缓冲(m)", "section_trim_buffer_m"),
            ("断面汇总表", "section_table_path"),
            ("临时目录", "temp_dir"),
            ("河流水系 CRS（可空）", "river_network_crs"),
            ("河流水系编码", "river_network_encoding"),
            ("道路 CRS（可空）", "road_crs"),
            ("道路编码", "road_encoding"),
        ]
        for idx, (label, key) in enumerate(labels):
            row = idx // 4 + 1
            col = (idx % 4) * 2
            ttk.Label(advanced, text=label).grid(row=row, column=col, sticky="w", padx=4, pady=3)
            ttk.Entry(advanced, textvariable=self._var(key), width=16).grid(row=row, column=col + 1, sticky="ew", padx=4, pady=3)
        self._add_path_row(advanced, 6, "河流水系（可选）", "river_network_path", "file_or_dir")
        self._add_path_row(advanced, 7, "道路/路网（可选）", "road_path", "file_or_dir")

        visual_frame = ttk.LabelFrame(self, text="动画显示区", padding=8)
        visual_frame.grid(row=3, column=0, sticky="nsew")
        visual_frame.columnconfigure(0, weight=1)
        visual_frame.rowconfigure(0, weight=1)
        self.map_view = FloodMapView(visual_frame)
        self.map_view.grid(row=0, column=0, sticky="nsew")

        button_frame = ttk.LabelFrame(self, text="运行控制区", padding=8)
        button_frame.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(button_frame, text="开始分析", command=self._start).pack(side=tk.LEFT, padx=6)
        ttk.Button(button_frame, text="清空参数", command=self._clear).pack(side=tk.LEFT, padx=6)
        ttk.Button(button_frame, text="打开输出文件夹", command=self._open_output).pack(side=tk.LEFT, padx=6)
        self.progress = ttk.Progressbar(button_frame, mode="indeterminate", length=180)
        self.progress.pack(side=tk.RIGHT, padx=6)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(button_frame, textvariable=self.status_var).pack(side=tk.RIGHT, padx=8)

    def _toggle_advanced(self) -> None:
        if self.advanced_visible.get():
            self.advanced_frame.grid_remove()
            self.advanced_button.configure(text="显示高级参数")
            self.advanced_visible.set(False)
        else:
            self.advanced_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
            self.advanced_button.configure(text="隐藏高级参数")
            self.advanced_visible.set(True)

    def _browse(self, key: str, mode: str) -> None:
        if mode == "dir":
            path = filedialog.askdirectory()
        elif mode == "multi_files":
            paths = filedialog.askopenfilenames(
                filetypes=[("断面/GIS/表格", "*.shp *.geojson *.gpkg *.xlsx *.xls *.csv *.dat *.DAT"), ("所有文件", "*.*")]
            )
            self._append_paths(key, paths)
            return
        elif mode == "append_dir":
            path = filedialog.askdirectory()
            if path:
                self._append_paths(key, [path])
            return
        elif mode == "file_or_dir":
            path = filedialog.askopenfilename()
            if not path:
                path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename(filetypes=[("GIS/表格/栅格", "*.shp *.geojson *.gpkg *.xlsx *.xls *.csv *.tif *.tiff *.out *.asc"), ("所有文件", "*.*")])
        if path:
            self._var(key).set(path)
            self._update_path_label(key)

    def _split_path_text(self, text: str) -> list[str]:
        return [item.strip() for item in text.replace("\n", ";").split(";") if item.strip()]

    def _append_paths(self, key: str, paths) -> None:
        existing = self._split_path_text(self._var(key).get())
        for path in paths:
            if path and path not in existing:
                existing.append(str(path))
        self._var(key).set(";".join(existing))
        self._update_path_label(key)

    def _path_display_text(self, text: str) -> str:
        paths = self._split_path_text(text)
        if not paths:
            return "未选择"
        if len(paths) == 1:
            return Path(paths[0]).name or paths[0]
        return f"已选择 {len(paths)} 个路径"

    def _update_path_label(self, key: str) -> None:
        if key in self.path_label_vars:
            self.path_label_vars[key].set(self._path_display_text(self._var(key).get()))

    def _collect_config(self) -> dict:
        cfg = {key: var.get().strip() for key, var in self.entries.items()}
        cfg["value_type"] = self.value_type.get()
        cfg["section_reference_mode"] = SECTION_REFERENCE_TO_VALUE.get(self.section_reference_mode.get(), self.section_reference_mode.get())
        if cfg.get("section_path"):
            cfg["section_paths"] = self._split_path_text(cfg["section_path"])
        cfg["use_temp_cropped_tif"] = True
        cfg["treat_zero_as_nodata"] = True
        cfg["stop_after_first_flood"] = True
        cfg["animate_all_hours"] = True
        cfg["all_touched"] = False
        if not cfg.get("scenario_name") and cfg.get("flood_folder"):
            cfg["scenario_name"] = Path(cfg["flood_folder"]).name
        if not cfg.get("threshold"):
            cfg["threshold"] = "0.1"
        if not cfg.get("time_interval_hours"):
            cfg["time_interval_hours"] = "1"
        if not cfg.get("section_buffer_m"):
            cfg["section_buffer_m"] = "5"
        if not cfg.get("max_section_distance_m"):
            cfg["max_section_distance_m"] = "3000"
        if not cfg.get("section_trim_mode"):
            cfg["section_trim_mode"] = "road_river"
        if not cfg.get("section_trim_buffer_m"):
            cfg["section_trim_buffer_m"] = "150"
        return cfg

    def _start(self) -> None:
        if self.running:
            messagebox.showinfo("正在运行", "分析任务正在运行，请等待完成。")
            return
        cfg = self._collect_config()
        self.status_var.set("正在分析...")
        self.progress.start(12)
        self.map_view.show_message("正在读取数据并计算淹没过程，请稍候。")
        self.running = True
        thread = threading.Thread(target=self._run_worker, args=(cfg,), daemon=True)
        thread.start()

    def _run_worker(self, cfg: dict) -> None:
        log_file = None
        try:
            output_dir = cfg.get("output_dir") or "outputs"
            log_file = Path(output_dir) / "logs" / "workflow.log"
            logger = WorkflowLogger(callback=lambda line: self.log_queue.put(("status", line)), log_file=log_file)
            result = run_analysis(cfg, logger)
            self.log_queue.put(("result", result))
        except Exception as exc:
            self.log_queue.put(("error", str(exc)))
        finally:
            self.running = False
            self.log_queue.put(("done", None))

    def _poll_logs(self) -> None:
        while True:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break
            kind, value = item if isinstance(item, tuple) else ("status", item)
            if kind == "status":
                self.status_var.set(str(value).split("]")[-1].strip())
            elif kind == "result":
                self.map_view.load_payload(value.map_payload)
                self.status_var.set(f"分析完成：{value.xlsx_path}")
            elif kind == "error":
                self.status_var.set(f"错误：{value}")
                self.map_view.show_message(f"分析失败：{value}")
                messagebox.showerror("分析失败", str(value))
            elif kind == "done":
                self.progress.stop()
        self.after(150, self._poll_logs)

    def _clear(self) -> None:
        for var in self.entries.values():
            var.set("")
        for key in self.path_label_vars:
            self._update_path_label(key)
        self.value_type.set("水深")
        self.section_reference_mode.set("首次受淹建筑物")
        self.map_view.show_message("开始分析后，这里会显示村庄建筑、附近断面和淹没过程动画。")
        self.status_var.set("就绪")

    def _open_output(self) -> None:
        path = self._var("output_dir").get().strip() or "outputs"
        Path(path).mkdir(parents=True, exist_ok=True)
        os.startfile(path)


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def create_app(config_path: Path | None = None) -> tk.Tk:
    config = load_config(config_path) if config_path else {}
    root = tk.Tk()
    FloodWorkflowApp(root, config)
    return root
