"""Regenerate service-dependent decision tables from the corrected V18 service closure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tespub.codesign_v16 import (  # noqa: E402
    advantage_transfer,
    decision_resolution_service_pareto,
    exact_service_pareto,
    load_config,
    pairwise_scale_order,
    pareto_precision_sensitivity,
)


def _key_set(frame: pd.DataFrame, columns: list[str]) -> set[tuple]:
    return set(frame[columns].itertuples(index=False, name=None))


def main() -> None:
    baseline = ROOT / "results_v16"
    output = ROOT / "results_v18"
    config = load_config(ROOT / "config" / "study_v18.json")
    unit = pd.read_csv(baseline / "codesign_results_v16.csv")
    service = pd.read_csv(output / "fixed_service_results_v18.csv")
    exact_front = exact_service_pareto(service, config)
    decision_front = decision_resolution_service_pareto(service, config)
    precision = pareto_precision_sensitivity(service, config)
    transfer = advantage_transfer(unit, service)
    pairwise = pairwise_scale_order(unit, service)
    tables = {
        "system_pareto_exact_audit_v18.csv": exact_front,
        "system_pareto_front_v18.csv": decision_front,
        "pareto_precision_sensitivity_v18.csv": precision,
        "advantage_transfer_v18.csv": transfer,
        "pairwise_scale_order_v18.csv": pairwise,
    }
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)

    old_service = pd.read_csv(baseline / "fixed_service_results_v16.csv")
    old_exact = pd.read_csv(baseline / "system_pareto_exact_audit_v16.csv")
    old_front = pd.read_csv(baseline / "system_pareto_front_v16.csv")
    old_pairwise = pd.read_csv(baseline / "pairwise_scale_order_v16.csv")
    service_keys = ["scenario_id", "design_id"]
    front_keys = ["scenario_id", "design_id"]
    pair_keys = [
        "scenario_id",
        "base_design_id",
        "topology_id",
        "fluid_model_id_i",
        "fluid_model_id_j",
    ]
    old_counts = old_service[service_keys + ["parallel_unit_count"]].rename(
        columns={"parallel_unit_count": "old_count"}
    )
    count_delta = service[service_keys + ["parallel_unit_count"]].merge(
        old_counts, on=service_keys, validate="one_to_one"
    )
    count_delta["delta"] = count_delta["parallel_unit_count"] - count_delta["old_count"]
    pair_compare = pairwise[pair_keys + ["system_exact_tie"]].merge(
        old_pairwise[pair_keys + ["system_exact_tie"]].rename(
            columns={"system_exact_tie": "old_system_exact_tie"}
        ),
        on=pair_keys,
        validate="one_to_one",
    )
    report = {
        "status": "PASS",
        "service_rows": int(len(service)),
        "count_changed_rows": int((count_delta["delta"] != 0).sum()),
        "count_increases": int((count_delta["delta"] > 0).sum()),
        "count_decreases": int((count_delta["delta"] < 0).sum()),
        "exact_front_rows_old": int(len(old_exact)),
        "exact_front_rows_v18": int(len(exact_front)),
        "exact_front_membership_changed": _key_set(old_exact, front_keys)
        != _key_set(exact_front, front_keys),
        "decision_front_rows_old": int(len(old_front)),
        "decision_front_rows_v18": int(len(decision_front)),
        "decision_front_membership_changed": _key_set(old_front, front_keys)
        != _key_set(decision_front, front_keys),
        "pairwise_rows": int(len(pairwise)),
        "old_exact_ties": int(old_pairwise["system_exact_tie"].astype(bool).sum()),
        "v18_exact_ties": int(pairwise["system_exact_tie"].astype(bool).sum()),
        "pairwise_tie_classification_changes": int(
            (
                pair_compare["system_exact_tie"].astype(bool)
                != pair_compare["old_system_exact_tie"].astype(bool)
            ).sum()
        ),
    }
    if len(service) != 7680 or len(pairwise) != 15360:
        report["status"] = "FAIL"
    (output / "00_service_dependent_rebuild_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("Service-dependent V18 rebuild failed")


if __name__ == "__main__":
    main()
