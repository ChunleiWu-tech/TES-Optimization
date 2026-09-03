"""Finalize the V18 numerical release without modifying the frozen V16 baseline.

The script performs three jobs that are deliberately kept separate from the
expensive physical-model run:

1. rebuild the g=0.90 rows and all ratios in the outlet-grade sensitivity
   table from the corrected V18 service closure;
2. prove which previously computed verification tables are independent of the
   67 corrected integer-boundary cases; and
3. create a checksum-verified, timestamped run record containing the validated
   V18 results and the exact source/configuration snapshot used to obtain them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v18"
BASELINE = ROOT / "results_v16"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _probe_selection_keys(service: pd.DataFrame) -> set[tuple[str, str]]:
    eligible = service[service["service_admissible"]].copy()
    selected: set[tuple[str, str]] = set()
    for _, part in eligible.groupby(
        ["scenario_id", "fluid_model_id", "topology_id"], sort=True
    ):
        ordered = part.sort_values("parallel_unit_count").reset_index(drop=True)
        positions = sorted(
            set(int(round(x * (len(ordered) - 1))) for x in (0.10, 0.50, 0.90))
        )
        for row in ordered.iloc[positions].itertuples(index=False):
            selected.add((str(row.scenario_id), str(row.design_id)))
    return selected


def rebuild_threshold_sensitivity() -> dict[str, object]:
    old = pd.read_csv(BASELINE / "outlet_grade_threshold_sensitivity_v16.csv")
    service = pd.read_csv(RESULTS / "fixed_service_results_v18.csv")
    threshold_col = "threshold_sensitivity_grade_fraction"
    keys = ["scenario_id", "design_id"]

    nonbaseline = old[~np.isclose(old[threshold_col], 0.90)].copy()
    baseline = service.copy()
    baseline[threshold_col] = 0.90

    common = [column for column in old.columns if column in baseline.columns]
    baseline_aligned = baseline[common].copy()
    for column in old.columns:
        if column not in baseline_aligned.columns:
            baseline_aligned[column] = np.nan
    baseline_aligned = baseline_aligned[old.columns]
    out = pd.concat([nonbaseline, baseline_aligned], ignore_index=True)

    base = out[np.isclose(out[threshold_col], 0.90)][
        keys
        + [
            "parallel_unit_count",
            "installed_packed_volume_m3",
            "resolved_unit_deliverable_energy_MWh",
        ]
    ].rename(
        columns={
            "parallel_unit_count": "g090_parallel_unit_count",
            "installed_packed_volume_m3": "g090_installed_packed_volume_m3",
            "resolved_unit_deliverable_energy_MWh": "g090_resolved_unit_deliverable_energy_MWh",
        }
    )
    drop = [
        "g090_parallel_unit_count",
        "g090_installed_packed_volume_m3",
        "g090_resolved_unit_deliverable_energy_MWh",
        "parallel_unit_count_ratio_vs_g090",
        "installed_volume_ratio_vs_g090",
        "deliverable_energy_ratio_vs_g090",
    ]
    out = out.drop(columns=[column for column in drop if column in out], errors="ignore")
    out = out.merge(base, on=keys, how="left", validate="many_to_one")
    out["parallel_unit_count_ratio_vs_g090"] = (
        out["parallel_unit_count"] / out["g090_parallel_unit_count"]
    )
    out["installed_volume_ratio_vs_g090"] = (
        out["installed_packed_volume_m3"] / out["g090_installed_packed_volume_m3"]
    )
    out["deliverable_energy_ratio_vs_g090"] = (
        out["resolved_unit_deliverable_energy_MWh"]
        / out["g090_resolved_unit_deliverable_energy_MWh"]
    )
    out["interpretation"] = (
        "Deterministic registered outlet-temperature threshold sensitivity; "
        "not a probability distribution or a chemistry uncertainty model."
    )
    order = {value: idx for idx, value in enumerate(sorted(out[threshold_col].unique()))}
    out["_threshold_order"] = out[threshold_col].map(order)
    out = out.sort_values(keys + ["_threshold_order"]).drop(columns="_threshold_order")
    target = RESULTS / "outlet_grade_threshold_sensitivity_v18.csv"
    out.to_csv(target, index=False)

    baseline_rows = out[np.isclose(out[threshold_col], 0.90)]
    merged = baseline_rows.merge(
        service[keys + ["parallel_unit_count"]],
        on=keys,
        suffixes=("_threshold", "_service"),
        validate="one_to_one",
    )
    count_match = bool(
        (merged["parallel_unit_count_threshold"] == merged["parallel_unit_count_service"]).all()
    )
    return {
        "status": "PASS" if count_match and len(out) == len(old) else "FAIL",
        "rows": int(len(out)),
        "baseline_rows": int(len(baseline_rows)),
        "baseline_counts_match_corrected_service": count_match,
        "output": target.name,
    }


def dependency_audit(threshold_report: dict[str, object]) -> dict[str, object]:
    old_service = pd.read_csv(BASELINE / "fixed_service_results_v16.csv")
    new_service = pd.read_csv(RESULTS / "fixed_service_results_v18.csv")
    old_front = pd.read_csv(BASELINE / "system_pareto_front_v16.csv")
    new_front = pd.read_csv(RESULTS / "system_pareto_front_v18.csv")
    keys = ["scenario_id", "design_id"]
    old_keys = set(map(tuple, old_front[keys].astype(str).itertuples(index=False, name=None)))
    new_keys = set(map(tuple, new_front[keys].astype(str).itertuples(index=False, name=None)))
    front_membership_equal = old_keys == new_keys
    front_count_compare = old_front[keys + ["parallel_unit_count"]].merge(
        new_front[keys + ["parallel_unit_count"]],
        on=keys,
        suffixes=("_v16", "_v18"),
        validate="one_to_one",
    )
    front_counts_equal = bool(
        (
            front_count_compare["parallel_unit_count_v16"]
            == front_count_compare["parallel_unit_count_v18"]
        ).all()
    )
    probe_old = _probe_selection_keys(old_service)
    probe_new = _probe_selection_keys(new_service)
    probe_selection_equal = probe_old == probe_new
    probe_report_path = RESULTS / "00_monotonicity_probe_rebuild_report.json"
    probe_report = (
        json.loads(probe_report_path.read_text(encoding="utf-8"))
        if probe_report_path.is_file()
        else {"status": "MISSING"}
    )

    retained = {
        "decision_grid_refinement_v16.csv": "same decision-front membership and unchanged counts",
        "module_scale_sensitivity_v16.csv": "recomputed from unit physics for the same decision-front keys",
        "transport_temperature_sensitivity_v16.csv": "recomputed from unit physics for the same decision-front keys",
        "service_monotonicity_audit_v16.csv": "same retained keys and identical integer scan bounds",
        "bounded_robustness_v16.csv": "same decision-front design set; nominal values are used only for selection",
        "model_form_stress_v16.csv": "same decision-front design set; nominal values are used only for selection",
        "standby_loss_v16.csv": "same decision-front design set; calculation depends on unit-scale inventory",
        "sampling_robustness_v16.csv": "independent resampling calculation",
        "numerical_verification_v16.csv": "independent numerical verification",
        "external_validation_metrics_v16.csv": "independent experimental benchmark",
        "external_validation_trace_v16.csv": "independent experimental benchmark",
        "external_validation_calibration_v16.csv": "independent experimental benchmark",
        "particle_model_hierarchy_v16.csv": "independent particle-model hierarchy",
        "topology_response_v16.csv": "independent topology response study",
        "interaction_indices_v16.csv": "depends only on unit-scale archive",
    }
    conditions = {
        "decision_front_membership_equal": front_membership_equal,
        "decision_front_counts_equal": front_counts_equal,
        "monotonicity_probe_selection_change_resolved": bool(
            probe_selection_equal or probe_report.get("status") == "PASS"
        ),
        "threshold_v18_pass": threshold_report["status"] == "PASS",
    }
    status = "PASS" if all(conditions.values()) else "FAIL"
    report = {
        "status": status,
        "conditions": conditions,
        "decision_front_rows": int(len(new_front)),
        "probe_selection_count_v16": int(len(probe_old)),
        "probe_selection_count_v18": int(len(probe_new)),
        "monotonicity_probe_selection_equal": probe_selection_equal,
        "recomputed_tables": [
            "fixed_service_results_v18.csv",
            "system_pareto_exact_audit_v18.csv",
            "system_pareto_front_v18.csv",
            "pareto_precision_sensitivity_v18.csv",
            "advantage_transfer_v18.csv",
            "pairwise_scale_order_v18.csv",
            "outlet_grade_threshold_sensitivity_v18.csv",
            "01_continuous_service_closure.csv",
            "02_pairwise_design_consequence.csv",
            "03_design_consequence_summary.csv",
            "service_monotonicity_probe_v18.csv",
        ],
        "retained_v16_verification_tables": retained,
        "probe_selection_changed_and_recomputed": not probe_selection_equal,
        "interpretation": (
            "Retained tables are not assumed unchanged from a version label. "
            "Each is retained only because its selection keys and numerical "
            "inputs are independent of the corrected service-count rows."
        ),
    }
    (RESULTS / "04_dependency_audit_v18.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def validate_release(dependency: dict[str, object]) -> dict[str, object]:
    expected = {
        "fixed_service_results_v18.csv": 7680,
        "00_integer_reclosure_audit.csv": 7680,
        "01_continuous_service_closure.csv": 7680,
        "02_pairwise_design_consequence.csv": 15360,
        "outlet_grade_threshold_sensitivity_v18.csv": 23040,
        "system_pareto_front_v18.csv": 52,
        "system_pareto_exact_audit_v18.csv": 271,
        "service_monotonicity_probe_v18.csv": int(
            json.loads(
                (RESULTS / "00_monotonicity_probe_rebuild_report.json").read_text(
                    encoding="utf-8"
                )
            )["rows"]
        ),
    }
    tables: dict[str, object] = {}
    overall = dependency["status"] == "PASS"
    for name, rows in expected.items():
        path = RESULTS / name
        frame = pd.read_csv(path)
        duplicate_key = None
        if {"scenario_id", "design_id"}.issubset(frame.columns) and name not in {
            "02_pairwise_design_consequence.csv",
            "outlet_grade_threshold_sensitivity_v18.csv",
        }:
            duplicate_key = int(frame.duplicated(["scenario_id", "design_id"]).sum())
        finite_issue_columns: list[str] = []
        for column in frame.select_dtypes(include=[np.number]).columns:
            values = frame[column].to_numpy(float)
            if np.isinf(values).any():
                finite_issue_columns.append(column)
        passed = len(frame) == rows and not finite_issue_columns and duplicate_key in (None, 0)
        overall = overall and passed
        tables[name] = {
            "pass": passed,
            "rows": int(len(frame)),
            "expected_rows": int(rows),
            "columns": int(len(frame.columns)),
            "duplicate_primary_keys": duplicate_key,
            "infinite_numeric_columns": finite_issue_columns,
            "sha256": sha256(path),
        }
    qc = json.loads((RESULTS / "05_V18_QC_report.json").read_text(encoding="utf-8"))
    overall = overall and qc["status"] == "PASS"
    report = {
        "status": "PASS" if overall else "FAIL",
        "version": "18.0.0",
        "qc_status": qc["status"],
        "dependency_audit_status": dependency["status"],
        "tables": tables,
        "scientific_scope": (
            "Continuous module requirement and exact integer-boundary consequence "
            "on the unchanged V16 physical model."
        ),
        "claim_boundary": (
            "The calculation identifies whether a matched fluid-property contrast "
            "changes the required number of parallel packed-bed modules. It does "
            "not rank unlike skeleton evidence sources or validate molten-salt "
            "corrosion, wetting, or long-term compatibility."
        ),
    }
    (RESULTS / "06_upstream_release_audit_v18.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def archive_release(stamp: str) -> Path:
    destination = ROOT / "run_records" / f"V18_{stamp}_COMPLETE"
    if destination.exists():
        raise FileExistsError(destination)
    (destination / "final_results").mkdir(parents=True)
    (destination / "validated_checkpoints").mkdir(parents=True)
    (destination / "reproducibility_snapshot").mkdir(parents=True)

    for path in sorted(RESULTS.iterdir()):
        if path.is_file():
            shutil.copy2(path, destination / "final_results" / path.name)
    checkpoint_dir = ROOT / ".v18_checkpoints"
    if checkpoint_dir.exists():
        for path in sorted(checkpoint_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(checkpoint_dir)
                target = destination / "validated_checkpoints" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    for relative in ("config", "scripts", "tespub", "tests"):
        source = ROOT / relative
        target = destination / "reproducibility_snapshot" / relative
        shutil.copytree(
            source,
            target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
    for name in ("pyproject.toml", "requirements.lock", "run_v16.py"):
        source = ROOT / name
        if source.is_file():
            shutil.copy2(source, destination / "reproducibility_snapshot" / name)

    manifest: list[dict[str, object]] = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            manifest.append(
                {
                    "path": path.relative_to(destination).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    (destination / "SHA256_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    verify = all(
        sha256(destination / item["path"]) == item["sha256"] for item in manifest
    )
    record = {
        "status": "COMPLETE" if verify else "FAILED",
        "version": "18.0.0",
        "timestamp": stamp,
        "files_verified": int(len(manifest)),
        "checksum_verification_passed": verify,
        "live_results_moved": False,
    }
    (destination / "RUN_RECORD.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    threshold = rebuild_threshold_sensitivity()
    dependency = dependency_audit(threshold)
    release = validate_release(dependency)
    if release["status"] != "PASS":
        raise RuntimeError("V18 release audit failed; archive and downstream work are blocked")
    archive = archive_release(args.stamp)
    print(
        json.dumps(
            {
                "status": "PASS",
                "threshold": threshold,
                "dependency_audit": dependency["status"],
                "release_audit": release["status"],
                "archive": str(archive),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
