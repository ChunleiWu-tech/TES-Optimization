"""Run the incremental V18 continuous-closure and design-consequence analysis."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tespub.design_consequence_v18 import (  # noqa: E402
    CONTINUOUS_FILENAME,
    PAIRWISE_FILENAME,
    SUMMARY_FILENAME,
    baseline_result_hashes,
    build_design_consequence_summary,
    build_pairwise_design_consequence,
    build_qc_report,
    load_config,
    run_continuous_service_closure,
    write_qc_report,
)


BASELINE = ROOT / "results_v16"
OUTPUT = ROOT / "results_v18"
CONFIG = ROOT / "config" / "study_v18.json"
HASH_MANIFEST = ROOT / "QA" / "V18_FROZEN_V16_BASELINE_HASHES.json"


def _baseline_names() -> list[str]:
    return sorted(path.name for path in BASELINE.iterdir() if path.is_file())


def _load_or_create_hash_manifest() -> dict[str, str]:
    names = _baseline_names()
    current = baseline_result_hashes(BASELINE, names)
    HASH_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if HASH_MANIFEST.is_file():
        saved = json.loads(HASH_MANIFEST.read_text(encoding="utf-8"))["sha256"]
        if saved != current:
            raise RuntimeError("Frozen V16 result hashes changed; V18 incremental analysis is blocked")
        return saved
    HASH_MANIFEST.write_text(
        json.dumps(
            {
                "scope": "All files in results_v16 at V18 initialization",
                "source": str(BASELINE),
                "sha256": current,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return current


def _copy_frozen_baseline() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for source in BASELINE.iterdir():
        if source.is_file():
            destination = OUTPUT / source.name
            if not destination.exists() or destination.stat().st_size != source.stat().st_size:
                shutil.copy2(source, destination)


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    unit = pd.read_csv(BASELINE / "codesign_results_v16.csv")
    corrected_service = OUTPUT / "fixed_service_results_v18.csv"
    service = pd.read_csv(
        corrected_service if corrected_service.is_file() else BASELINE / "fixed_service_results_v16.csv"
    )
    pairwise = pd.read_csv(BASELINE / "pairwise_scale_order_v16.csv")
    config = load_config(CONFIG)
    if len(unit) != 3840 or len(service) != 7680 or len(pairwise) != 15360:
        raise RuntimeError(
            f"Unexpected frozen baseline sizes: unit={len(unit)}, service={len(service)}, pairwise={len(pairwise)}"
        )
    return unit, service, pairwise, config


def run_qc(workers: int | None) -> None:
    unit, service, _, config = _load_inputs()
    qc_dir = ROOT / "QA" / "v18_fractional_count_qc"
    result = run_continuous_service_closure(
        unit, service, config, qc_dir, qc_only=True, workers=workers
    )
    residual_limit = float(config["design_consequence_boundary"]["root_relative_energy_tolerance"])
    passed = bool(
        result["N_star_match"].all()
        and (result["root_relative_residual"] <= residual_limit).all()
        and result["within_registered_count_and_duty_limits"].all()
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "n_cases": int(len(result)),
        "n_integer_matches": int(result["N_star_match"].sum()),
        "maximum_root_relative_residual": float(result["root_relative_residual"].max()),
        "registered_root_relative_residual_limit": residual_limit,
        "n_within_registered_count_and_duty_limits": int(
            result["within_registered_count_and_duty_limits"].sum()
        ),
        "services": sorted(result["scenario_id"].unique().tolist()),
        "fluids": sorted(result["fluid_model_id"].unique().tolist()),
        "old_constraint_classes": sorted(result["old_governing_constraint"].unique().tolist()),
    }
    (qc_dir / "V18_FRACTIONAL_COUNT_QC.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("V18 fractional-count QC failed")


def run_full(workers: int | None) -> None:
    frozen_hashes = _load_or_create_hash_manifest()
    _copy_frozen_baseline()
    unit, service, baseline_pairwise, config = _load_inputs()
    continuous = run_continuous_service_closure(
        unit, service, config, OUTPUT, qc_only=False, workers=workers
    )
    pairwise = build_pairwise_design_consequence(
        unit, service, continuous, baseline_pairwise, config
    )
    pairwise.to_csv(OUTPUT / PAIRWISE_FILENAME, index=False)
    summary = build_design_consequence_summary(pairwise)
    summary.to_csv(OUTPUT / SUMMARY_FILENAME, index=False)
    current_hashes = baseline_result_hashes(BASELINE, sorted(frozen_hashes))
    report = build_qc_report(
        continuous,
        pairwise,
        service,
        baseline_pairwise,
        frozen_hashes,
        current_hashes,
        config,
    )
    correction_report_path = OUTPUT / "00_service_correction_report.json"
    dependent_report_path = OUTPUT / "00_service_dependent_rebuild_report.json"
    if correction_report_path.is_file() and dependent_report_path.is_file():
        correction_report = json.loads(correction_report_path.read_text(encoding="utf-8"))
        dependent_report = json.loads(dependent_report_path.read_text(encoding="utf-8"))
        correction_pass = bool(
            correction_report.get("status") == "PASS"
            and correction_report.get("all_corrected_rows_match_independent_integer_audit")
            and dependent_report.get("status") == "PASS"
            and not dependent_report.get("decision_front_membership_changed", True)
            and int(dependent_report.get("pairwise_tie_classification_changes", -1)) == 0
        )
        report["gates"] = {
            "Q0_integer_reclosure_and_dependency_regression": {
                "pass": correction_pass,
                "corrected_count_rows": int(correction_report["corrected_count_rows"]),
                "count_increases": int(correction_report["count_increases"]),
                "count_decreases": int(correction_report["count_decreases"]),
                "maximum_absolute_frozen_capacity_reproduction_error_pct": float(
                    correction_report[
                        "maximum_absolute_frozen_capacity_reproduction_error_pct"
                    ]
                ),
                "decision_front_membership_changed": bool(
                    dependent_report["decision_front_membership_changed"]
                ),
                "exact_front_rows_old": int(dependent_report["exact_front_rows_old"]),
                "exact_front_rows_v18": int(dependent_report["exact_front_rows_v18"]),
                "pairwise_tie_classification_changes": int(
                    dependent_report["pairwise_tie_classification_changes"]
                ),
            },
            **report["gates"],
        }
        report["status"] = (
            "PASS"
            if correction_pass and all(gate["pass"] for gate in report["gates"].values())
            else "FAIL"
        )
    write_qc_report(report, OUTPUT)
    run_summary = {
        "status": report["status"],
        "version": "18.0.0",
        "incremental_outputs": [CONTINUOUS_FILENAME, PAIRWISE_FILENAME, SUMMARY_FILENAME],
        "n_continuous_cases": int(len(continuous)),
        "n_pairwise_cases": int(len(pairwise)),
        "n_configuration_equivalent": int(pairwise["same_integer"].sum()),
        "n_groups_with_equivalence": int(pairwise.groupby("group_id")["same_integer"].any().sum()),
        "configuration_equivalent_fraction": float(pairwise["same_integer"].mean()),
        "maximum_root_relative_residual": float(continuous["root_relative_residual"].max()),
        "baseline_version": "16.0.0",
        "physical_model_changed": False,
        "service_integer_reclosure_applied": bool(correction_report_path.is_file()),
        "corrected_service_count_rows": (
            int(correction_report["corrected_count_rows"])
            if correction_report_path.is_file() and dependent_report_path.is_file()
            else 0
        ),
        "analysis_coordinate_added": "continuous module requirement and exact integer-boundary margin",
    }
    (OUTPUT / "run_summary_v18.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_summary, indent=2, ensure_ascii=False), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("V18 acceptance gates failed; manuscript update is blocked")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("qc", "full"), default="qc")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "qc":
        run_qc(args.workers)
    else:
        run_full(args.workers)


if __name__ == "__main__":
    main()
