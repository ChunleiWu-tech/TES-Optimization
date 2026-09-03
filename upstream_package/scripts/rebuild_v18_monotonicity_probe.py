"""Rebuild the stratified V18 integer-capacity monotonicity probe in parallel."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tespub.codesign_v16 import _service_capacity_scan


def _one_probe(
    task: tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[int]]
) -> pd.DataFrame:
    source, scenario, config, counts = task
    frame = _service_capacity_scan(source, scenario, config, counts)
    frame["probe_kind"] = "stratified_log_count_counterexample_search"
    upper = int(max(counts))
    frame["reported_minimum_unit_count"] = upper
    delta = frame["installed_deliverable_energy_MWh"].diff()
    frame["capacity_increment_MWh"] = delta
    frame["nondecreasing_from_previous_count"] = delta.isna() | (delta >= -1e-10)
    return frame


def build_tasks(
    unit: pd.DataFrame, service: pd.DataFrame, config: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[int]]]:
    source_by_id = unit.set_index("design_id", drop=False)
    scenario_by_id = {row["scenario_id"]: row for row in config["service_scenarios"]}
    eligible = service[service["service_admissible"]].copy()
    selections: list[dict[str, Any]] = []
    for _, part in eligible.groupby(
        ["scenario_id", "fluid_model_id", "topology_id"], sort=True
    ):
        ordered = part.sort_values("parallel_unit_count").reset_index(drop=True)
        positions = sorted(
            set(int(round(x * (len(ordered) - 1))) for x in (0.10, 0.50, 0.90))
        )
        selections.extend(ordered.iloc[positions].to_dict("records"))
    tasks = []
    for selected in selections:
        lower = int(selected["power_limited_unit_count"])
        upper = int(selected["parallel_unit_count"])
        if upper <= lower:
            counts = [lower]
        else:
            counts = np.unique(
                np.rint(np.geomspace(lower, upper, num=9)).astype(int)
            ).tolist()
            counts = sorted(set([lower, *counts, upper]))
        tasks.append(
            (
                source_by_id.loc[str(selected["design_id"])].to_dict(),
                scenario_by_id[str(selected["scenario_id"])],
                config,
                counts,
            )
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    config = json.loads((ROOT / "config" / "study_v18.json").read_text(encoding="utf-8"))
    unit = pd.read_csv(ROOT / "results_v18" / "codesign_results_v16.csv")
    service = pd.read_csv(ROOT / "results_v18" / "fixed_service_results_v18.csv")
    tasks = build_tasks(unit, service, config)
    frames: list[pd.DataFrame | None] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(_one_probe, task): idx for idx, task in enumerate(tasks)}
        done = 0
        for future in as_completed(future_map):
            idx = future_map[future]
            frames[idx] = future.result()
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"[V18 monotonicity probe] {done}/{len(tasks)} selections", flush=True)
    out = pd.concat([frame for frame in frames if frame is not None], ignore_index=True)
    target = ROOT / "results_v18" / "service_monotonicity_probe_v18.csv"
    out.to_csv(target, index=False)
    passed = bool(out["nondecreasing_from_previous_count"].all())
    report = {
        "status": "PASS" if passed else "FAIL",
        "selection_cases": int(len(tasks)),
        "rows": int(len(out)),
        "all_local_capacity_sequences_nondecreasing": passed,
        "workers": int(args.workers),
        "output": target.name,
    }
    (ROOT / "results_v18" / "00_monotonicity_probe_rebuild_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise RuntimeError("V18 monotonicity probe found a local non-monotone sequence")


if __name__ == "__main__":
    main()
