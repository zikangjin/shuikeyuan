from __future__ import annotations

import argparse
from pathlib import Path

from src.flood_analysis import run_analysis
from src.gui import create_app, load_config
from src.logger_utils import WorkflowLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="村庄淹没与最近断面 Windows 桌面工具")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--no-gui", action="store_true", help="不打开窗口，直接按配置运行分析")
    parser.add_argument("--check-gui", action="store_true", help="创建并销毁窗口，用于检查 tkinter 是否可用")
    args = parser.parse_args()                                                                           

    config_path = Path(args.config)
    if args.no_gui:
        config = load_config(config_path)
        log_dir = Path(config.get("output_dir") or "outputs") / "logs"
        logger = WorkflowLogger(log_file=log_dir / "workflow.log")
        run_analysis(config, logger)
        return

    root = create_app(config_path)
    if args.check_gui:
        root.update()
        root.destroy()
        print("GUI check ok")
        return
    root.mainloop()


if __name__ == "__main__":
    main()
