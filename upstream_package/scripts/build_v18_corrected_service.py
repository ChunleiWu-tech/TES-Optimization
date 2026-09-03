"""Build a V18 service table corrected by the full integer reclosure audit."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tespub.codesign_v16 import _size_fixed_service_record, load_config  # noqa: E402


BASELINE = ROOT / "results_v16"
RESULTS = ROOT / "results_v18"
CONFIG = ROOT / "config" / "study_v18.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    unit = pd.read_csv(BASELINE / "codesign_results_v16.csv")
    frozen = pd.read_csv(BASELINE / "fixed_service_results_v16.csv")
    audit = pd.read_csv(RESULTS / "00_integer_reclosure_audit.csv")
    config = load_config(CONFIG)
    keys = ["scenario_id", "design_id"]
    if len(audit) != len(frozen) or audit[keys].duplicated().any():
        raise RuntimeError("Integer reclosure audit is incomplete or non-unique")
    if (
        audit["lower_search_truncated"].astype(bool).any()
        or audit["upper_search_truncated"].astype(bool).any()
        or (~audit["local_integer_capacity_nondecreasing"].astype(bool)).any()
    ):
        raise RuntimeError("Integer reclosure audit did not establish a local minimum")

    merged = frozen.merge(
        audit[
            keys
            + [
                "current_minimum_count",
                "corrected_installed_energy_MWh",
                "count_delta_vs_frozen",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    corrected = frozen.copy()
    current_count = merged["current_minimum_count"].astype(int)
    current_energy = merged["corrected_installed_energy_MWh"].astype(float)
    required = corrected["required_energy_MWh"].astype(float)
    corrected["parallel_unit_count"] = current_count
    corrected["installed_high_grade_capacity_MWh"] = current_energy
    corrected["resolved_unit_deliverable_energy_MWh"] = current_energy / current_count
    corrected["service_capacity_closure_error_pct"] = 100.0 * (current_energy / required - 1.0)
    corrected["capacity_oversize_pct"] = corrected["service_capacity_closure_error_pct"]
    corrected["resolved_high_grade_duration_h"] = (
        current_energy / corrected["target_thermal_power_MW"].astype(float)
    )
    corrected["resolved_unit_mass_flow_kg_s"] = (
        frozen["resolved_unit_mass_flow_kg_s"].astype(float)
        * frozen["parallel_unit_count"].astype(float)
        / current_count.astype(float)
    )
    corrected["resolved_unit_thermal_power_kW"] = (
        corrected["target_thermal_power_MW"].astype(float) * 1000.0 / current_count
    )
    corrected["installed_packed_volume_m3"] = (
        frozen["installed_packed_volume_m3"].astype(float)
        / frozen["parallel_unit_count"].astype(float)
        * current_count
    )
    corrected["salt_inventory_t"] = (
        frozen["salt_inventory_t"].astype(float)
        / frozen["parallel_unit_count"].astype(float)
        * current_count
    )
    corrected["solid_inventory_t"] = (
        frozen["solid_inventory_t"].astype(float)
        / frozen["parallel_unit_count"].astype(float)
        * current_count
    )
    corrected["service_compactness_MWh_m3"] = (
        required / corrected["installed_packed_volume_m3"].astype(float)
    )

    changed_audit = audit[audit["count_delta_vs_frozen"] != 0].copy()
    unit_index = unit.set_index("design_id", drop=False)
    scenarios = {str(row["scenario_id"]): row for row in config["service_scenarios"]}
    tasks = [
        (
            unit_index.loc[str(row["design_id"])].to_dict(),
            scenarios[str(row["scenario_id"])],
            config,
        )
        for row in changed_audit.to_dict("records")
    ]
    if int(args.workers) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            changed_rows = list(executor.map(_size_fixed_service_record, tasks, chunksize=1))
    else:
        changed_rows = [_size_fixed_service_record(task) for task in tasks]
    changed = pd.DataFrame(changed_rows)
    expected = changed_audit.set_index(keys)["current_minimum_count"].astype(int)
    observed = changed.set_index(keys)["parallel_unit_count"].astype(int)
    if not observed.equals(expected.loc[observed.index]):
        mismatch = pd.concat(
            [observed.rename("observed"), expected.rename("expected")], axis=1
        ).query("observed != expected")
        raise RuntimeError(f"Corrected service rows disagree with integer audit:\n{mismatch}")
    corrected_index = corrected.set_index(keys)
    changed_index = changed.set_index(keys)
    corrected_index.loc[changed_index.index, changed_index.columns] = changed_index
    corrected = corrected_index.reset_index()[frozen.columns]
    corrected = corrected.sort_values(keys).reset_index(drop=True)
    destination = RESULTS / "fixed_service_results_v18.csv"
    corrected.to_csv(destination, index=False)

    report = {
        "status": "PASS",
        "rows": int(len(corrected)),
        "corrected_count_rows": int(len(changed)),
        "count_decreases": int((changed_audit["count_delta_vs_frozen"] < 0).sum()),
        "count_increases": int((changed_audit["count_delta_vs_frozen"] > 0).sum()),
        "maximum_absolute_count_change": int(
            changed_audit["count_delta_vs_frozen"].abs().max()
        ),
        "maximum_absolute_frozen_capacity_reproduction_error_pct": float(
            audit["frozen_capacity_reproduction_error_pct"].abs().max()
        ),
        "all_corrected_rows_match_independent_integer_audit": True,
        "unchanged_count_rows": int((audit["count_delta_vs_frozen"] == 0).sum()),
        "results_v16_modified": False,
    }
    (RESULTS / "00_service_correction_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
