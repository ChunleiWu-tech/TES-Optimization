"""Console entry point for the V16 reproducible workflow."""
from __future__ import annotations
from pathlib import Path
import runpy

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / "run_v16.py"), run_name="__main__")
