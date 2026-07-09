from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable


class WorkflowLogger:
    """Small logger that writes to a callback and an optional file."""

    def __init__(self, callback: Callable[[str], None] | None = None, log_file: str | Path | None = None):
        self.callback = callback
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def warning(self, message: str) -> None:
        self._write("WARNING", message)

    def error(self, message: str) -> None:
        self._write("ERROR", message)

    def _write(self, level: str, message: str) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {level} {message}"
        if self.callback:
            self.callback(line)
        else:
            print(line)
        if self.log_file:
            encoding = "utf-8-sig" if not self.log_file.exists() or self.log_file.stat().st_size == 0 else "utf-8"
            with self.log_file.open("a", encoding=encoding) as f:
                f.write(line + "\n")
