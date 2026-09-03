"""Run V16 tests and write a machine-readable release report."""
from __future__ import annotations

import io
import json
import time
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_v16.py")
    stream = io.StringIO()
    start = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    elapsed = time.perf_counter() - start
    report = {
        "version": "16.0.0",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "elapsed_s": elapsed,
        "transcript": stream.getvalue().splitlines(),
    }
    out = ROOT / "results_v16" / "unit_test_report_v16.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("version", "status", "tests_run", "failures", "errors", "skipped", "elapsed_s")}, indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
