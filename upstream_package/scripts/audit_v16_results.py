"""Fail-closed structural, numerical and claim-boundary audit for V16."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results_v16"


def main() -> None:
    findings: list[dict[str, object]] = []
    def check(ok: bool, name: str, detail: object | None = None) -> None:
        findings.append({"check": name, "pass": bool(ok), "detail": detail})

    unit = pd.read_csv(R / "codesign_results_v16.csv")
    screen = pd.read_csv(R / "screen_to_confirmation_v16.csv")
    ufront = pd.read_csv(R / "unit_pareto_front_v16.csv")
    service = pd.read_csv(R / "fixed_service_results_v16.csv")
    sfront = pd.read_csv(R / "system_pareto_front_v16.csv")
    exact_front = pd.read_csv(R / "system_pareto_exact_audit_v16.csv")
    validation = pd.read_csv(R / "external_validation_metrics_v16.csv")
    numerical = pd.read_csv(R / "numerical_verification_v16.csv")
    particles = pd.read_csv(R / "particle_model_hierarchy_v16.csv")
    robust = pd.read_csv(R / "bounded_robustness_v16.csv")
    sampling = pd.read_csv(R / "sampling_robustness_v16.csv")
    precision = pd.read_csv(R / "pareto_precision_sensitivity_v16.csv")
    standby = pd.read_csv(R / "standby_loss_v16.csv")
    model_stress = pd.read_csv(R / "model_form_stress_v16.csv")
    pairwise = pd.read_csv(R / "pairwise_scale_order_v16.csv")
    threshold = pd.read_csv(R / "outlet_grade_threshold_sensitivity_v16.csv")
    module_scale = pd.read_csv(R / "module_scale_sensitivity_v16.csv")
    transport = pd.read_csv(R / "transport_temperature_sensitivity_v16.csv")
    monotonicity = pd.read_csv(R / "service_monotonicity_audit_v16.csv")
    monotonicity_probe = pd.read_csv(R / "service_monotonicity_probe_v16.csv")
    grid_refinement = pd.read_csv(R / "decision_grid_refinement_v16.csv")
    pair = pd.read_csv(ROOT / "data" / "salt_skeleton_pair_registry.csv")
    summary = json.loads((R / "analysis_summary_v16.json").read_text(encoding="utf-8"))

    check(len(unit) == 3840, "256 geometries × 3 topologies × 5 fluids", len(unit))
    check(unit.design_id.is_unique, "unit design IDs unique")
    check(set(unit.n_cells) == {2048}, "all published unit results use 2048-cell confirmation", sorted(unit.n_cells.unique()))
    check(len(screen) == len(unit), "screen-to-confirmation mapping complete", len(screen))
    check(screen.charge_t90_h_relative_difference_pct.abs().quantile(.95) < 2.0, "screen t90 p95 difference below 2%", float(screen.charge_t90_h_relative_difference_pct.abs().quantile(.95)))
    check(screen.high_grade_discharge_fraction_absolute_difference.abs().quantile(.95) < .08, "screen high-grade p95 absolute difference below 0.08", float(screen.high_grade_discharge_fraction_absolute_difference.abs().quantile(.95)))
    check(not unit.temperature_clipping_used.astype(bool).any(), "temperature clipping never used")
    check(unit.cycle_coupled_discharge.astype(bool).all(), "every unit discharge inherits its simulated charge-end profile")
    check((unit.charge_final_inventory_fraction > 0.99).all(), "charge horizon prepares at least 99% of rated inventory before discharge", float(unit.charge_final_inventory_fraction.min()))
    check(unit.charge_energy_balance_error_pct.abs().max() < 1e-6, "charge discrete energy balance closes", float(unit.charge_energy_balance_error_pct.abs().max()))
    check(unit.discharge_energy_balance_error_pct.abs().max() < 1e-6, "discharge discrete energy balance closes", float(unit.discharge_energy_balance_error_pct.abs().max()))
    check((unit.temperature_min_K >= 773.15 - 1e-6).all() and (unit.temperature_max_K <= 823.15 + 1e-6).all(), "bounded temperatures without clipping", [float(unit.temperature_min_K.min()), float(unit.temperature_max_K.max())])
    check(unit.temperature_extrema_scope.eq("initial state and every explicit transient time step").all(), "reported temperature extrema cover the complete transient", unit.temperature_extrema_scope.value_counts().to_dict())
    front_with_local_interface_step = unit.thermal_front_nonmonotonic_steps.gt(0)
    check(
        (unit.thermal_front_crossings_90 == 1).all()
        and (unit.thermal_front_crossings_10 == 1).all()
        and unit.thermal_front_nonmonotonic_steps.le(1).all()
        and unit.loc[front_with_local_interface_step, "topology"].eq("graded_two_layer").all(),
        "reported thermocline thicknesses have unique 0.90 and 0.10 crossings; any one-cell local departure is confined to the graded-layer interface",
        {
            "crossings_90": sorted(unit.thermal_front_crossings_90.unique().tolist()),
            "crossings_10": sorted(unit.thermal_front_crossings_10.unique().tolist()),
            "maximum_nonmonotonic_steps": int(unit.thermal_front_nonmonotonic_steps.max()),
            "interface_local_departures": int(front_with_local_interface_step.sum()),
            "interface_topologies": sorted(unit.loc[front_with_local_interface_step, "topology"].unique().tolist()),
        },
    )
    check(unit.physics_comparison_admissible.astype(bool).all(), "all baseline physics comparisons pass registered unit gates")
    check(not unit.engineering_deployment_eligible.astype(bool).any(), "no thermal comparison is promoted to deployment eligibility")
    check(pair.physics_comparison_eligible.astype(str).str.lower().eq("true").all(), "all five salt–skeleton pairs eligible only for physics comparison")
    check(pair.engineering_deployment_eligible.astype(str).str.lower().eq("false").all(), "all five pair deployment gates remain closed")
    check(len(ufront) > 0 and set(ufront.design_id).issubset(set(unit.design_id)), "unit Pareto front non-empty and traceable", len(ufront))
    check(len(service) == 7680 and service.design_id.notna().all(), "two fixed-service scenarios complete", len(service))
    check(service.parallel_unit_count.ge(1).all() and np.allclose(service.parallel_unit_count, np.round(service.parallel_unit_count)), "discrete modular unit counts valid")
    check(service.service_capacity_closure_error_pct.min() >= -1e-8, "deliverable service energy closes at resolved flow", float(service.service_capacity_closure_error_pct.min()))
    check(not service.adjacent_lower_count_feasible.astype(bool).any(), "adjacent lower integer unit count is infeasible")
    check(np.allclose(service.parallel_unit_count * service.resolved_unit_deliverable_energy_MWh, service.installed_high_grade_capacity_MWh), "installed capacity uses threshold-defined deliverable energy")
    check(service.cycle_coupled_discharge.astype(bool).all(), "every service closure retains cycle-coupled discharge")
    check(service.threshold_crossing_interpolation_used.astype(bool).all(), "every service result uses within-step threshold interpolation")
    check(np.isfinite(service.charge_discharge_t90_asymmetry_pct).all(), "charge-discharge asymmetry is resolved")
    check(service.service_admissible.astype(bool).all(), "all registered baseline fixed-service cases meet model-scope constraints")
    check(service.header_pressure_drop_kPa.gt(0).all(), "geometry-normalized on-array header friction is resolved for every service case")
    check(service.header_hydraulic_model.str.contains("Darcy-Weisbach", regex=False).all(), "hydraulic screen records its Darcy-Weisbach scope")
    check(service.pump_energy_definition.str.contains("fittings", case=False).all(), "hydraulic objective states excluded installed-pump components")
    check(not service.engineering_deployment_eligible.astype(bool).any(), "no model-admissible service case is promoted to deployment eligibility")
    check(len(pairwise) == 15360, "all 1,536 matched groups contain ten fluid pairs", len(pairwise))
    check((pairwise.pair_fate == "reversal").sum() == 0, "no matched fluid pair reverses at system scale")
    check((pairwise.pair_fate == "system_exact_tie").sum() > 0, "integer closure produces at least one exact array tie from a nonzero material difference")
    tie_groups = pairwise.groupby(["scenario_id", "base_design_id", "topology_id"]).system_exact_tie.any()
    check(int(tie_groups.sum()) + int((~tie_groups).sum()) == 1536, "tie and fully strict groups exhaust the 1,536 matched groups", {"with_ties": int(tie_groups.sum()), "fully_strict": int((~tie_groups).sum())})
    power_limited = service[service.power_limited_unit_count > service.resolved_energy_limited_unit_count]
    energy_limited = service.drop(power_limited.index)
    energy_count_delta = service.resolved_energy_limited_unit_count - service.nominal_energy_limited_unit_count
    check((energy_count_delta <= 0).all(), "flow re-solution never increases the energy-limited integer", {"increased_cases": int((energy_count_delta > 0).sum())})
    check((energy_count_delta < 0).any() and (energy_count_delta == 0).any(), "flow re-solution records both decreased and unchanged energy-limited integers", {"decreased_cases": int((energy_count_delta < 0).sum()), "unchanged_cases": int((energy_count_delta == 0).sum())})
    check(0 < len(power_limited) < len(service), "power-limited and energy-limited service cases are both present", {"power_limited": len(power_limited), "energy_limited": len(energy_limited)})
    check(power_limited.capacity_oversize_pct.median() > energy_limited.capacity_oversize_pct.median(), "power-limited excess capacity is separated from energy-limited integer closure")
    check(energy_limited.capacity_oversize_pct.min() >= -1e-8, "energy-limited integer oversize is nonnegative")
    check(len(sfront) > 0 and set(sfront.design_id).issubset(set(service.design_id)), "system Pareto front non-empty and traceable", len(sfront))
    check(len(exact_front) >= len(sfront), "decision-resolution front does not exceed exact audit front", {"exact": len(exact_front), "decision": len(sfront)})
    check(set(precision.decision_precision_multiplier) == {0.0, 0.5, 1.0, 2.0}, "Pareto precision sensitivity complete")
    check(
        len(grid_refinement) == len(sfront)
        and set(grid_refinement.production_n_cells) == {2048}
        and set(grid_refinement.refinement_n_cells) == {4096}
        and grid_refinement.grid_refinement_accepted.astype(bool).all(),
        "every retained decision passes the 4096-cell release gate",
        {"retained": len(grid_refinement), "accepted": int(grid_refinement.grid_refinement_accepted.astype(bool).sum())},
    )
    check(grid_refinement.deliverable_energy_change_pct.abs().max() <= 0.5, "retained decision deliverable-energy change remains within 0.5%", float(grid_refinement.deliverable_energy_change_pct.abs().max()))
    check(grid_refinement.high_grade_fraction_change.abs().max() <= 0.005, "retained decision high-grade fraction change remains within 0.005", float(grid_refinement.high_grade_fraction_change.abs().max()))
    check(set(validation.role) == {"calibration", "validation", "diagnostic"}, "external validation roles separated")
    hold = validation.loc[validation.role == "validation"].iloc[0]
    check("held out" in hold.claim_boundary.lower(), "cooling segment explicitly retained as holdout")
    diag = validation.loc[validation.role == "diagnostic"].iloc[0]
    check("figure-digitized" in diag.claim_boundary.lower(), "external benchmark digitisation boundary explicit")
    adv = numerical[numerical.verification_case == "periodic_step_advection"]
    if len(adv) >= 2:
        muscl = adv[adv.method.str.contains("MUSCL")].iloc[0]
        first = adv[adv.method.str.contains("first", case=False)].iloc[0]
        check(muscl.L1_error < first.L1_error and muscl.transition_cells_0p1_to_0p9 < first.transition_cells_0p1_to_0p9, "MUSCL reduces numerical diffusion", {"MUSCL_L1": float(muscl.L1_error), "first_order_L1": float(first.L1_error)})
    else:
        check(False, "MUSCL numerical-diffusion comparison present")
    check(particles.effective_resistance_error_vs_radial_pct.abs().max() < 3.0, "effective particle resistance agrees with radial FV within 3%", float(particles.effective_resistance_error_vs_radial_pct.abs().max()))
    check(particles.lumped_error_vs_radial_pct.abs().max() > 50.0, "lumped particle model demonstrably invalid at high Bi", float(particles.lumped_error_vs_radial_pct.abs().max()))
    time = numerical[numerical.verification_case == "time_step_convergence"]
    check(len(time) == 3 and time.t90_relative_error_vs_finest_pct.abs().max() < 2.0, "time-step sensitivity resolved", float(time.t90_relative_error_vs_finest_pct.abs().max()))
    check(time.high_grade_absolute_error_vs_finest.abs().max() < 0.01, "threshold-qualified energy time-step sensitivity remains below 0.01 absolute fraction", float(time.high_grade_absolute_error_vs_finest.abs().max()))
    check(
        len(threshold) == 23040
        and set(np.round(threshold.threshold_sensitivity_grade_fraction, 2)) == {0.85, 0.90, 0.95}
        and threshold.threshold_crossing_interpolation_used.astype(bool).all(),
        "all 7,680 service cases are re-closed at g = 0.85, 0.90 and 0.95",
        len(threshold),
    )
    threshold_medians = threshold.groupby("threshold_sensitivity_grade_fraction").parallel_unit_count_ratio_vs_g090.median()
    check(
        threshold_medians.loc[0.85] <= threshold_medians.loc[0.90] <= threshold_medians.loc[0.95],
        "median required count responds monotonically to the outlet-grade threshold",
        threshold_medians.to_dict(),
    )
    check(
        len(module_scale) == len(sfront) * 3
        and set(module_scale.linear_scale_factor) == {1.0, 2.0, 4.0}
        and module_scale.extrapolative_scale_test.sum() == len(sfront) * 2,
        "retained designs include explicit reference-to-module scale sensitivity",
        len(module_scale),
    )
    check(
        len(transport) == len(sfront) * 3
        and set(transport.transport_temperature_case) == {"cold_endpoint", "mean", "hot_endpoint"},
        "run-wise frozen transport properties are bounded at cold, mean and hot duty temperatures",
        len(transport),
    )
    monotonic_cases = monotonicity.groupby(["scenario_id", "reference_design_id"]).agg(
        nondecreasing=("nondecreasing_from_previous_count", "all"),
        minimum_matches=("reported_minimum_matches_full_scan", "all"),
    )
    check(
        len(monotonic_cases) == len(sfront)
        and monotonic_cases.nondecreasing.all()
        and monotonic_cases.minimum_matches.all(),
        "full integer scans confirm monotone installed energy and the reported minima for every retained decision case",
        {"cases": len(monotonic_cases), "counts_scanned": len(monotonicity)},
    )
    probe_cases = monotonicity_probe.groupby(["scenario_id", "reference_design_id"])["nondecreasing_from_previous_count"].all()
    check(len(probe_cases) > 0 and probe_cases.all(), "stratified counterexample search finds no non-monotone sampled capacity increment", {"cases": len(probe_cases), "evaluations": len(monotonicity_probe)})
    check(
        summary["validation_threshold_context"]["RMSE_to_threshold_margin_ratio"] > 1.0
        and "not transferred" in summary["validation_threshold_context"]["interpretation"].lower(),
        "air-bed cooling discrepancy is compared with the threshold margin but not transferred as molten-salt uncertainty",
        summary["validation_threshold_context"],
    )
    check((robust.design_id == "NOT_ASSESSED").sum() == 4, "missing property uncertainty is retained as not assessed", int((robust.design_id == "NOT_ASSESSED").sum()))
    check(len(sampling) == 24 and set(sampling.n_base_geometries) == {64, 128, 192} and sampling.seed.nunique() == 4, "independent sampling sensitivity covers four seeds and both services")
    check(set(robust.scenario_id) == {"S10MW_8H", "S50MW_6H"}, "property stress covers both services")
    check(set(standby.standby_duration_h) == {2.0, 6.0, 12.0, 24.0}, "standby-loss durations complete")
    check(set(model_stress.stress_id) == {"intrinsic_adiabatic", "bounded_moderate", "bounded_severe"}, "model-form stress hierarchy complete")
    check(summary.get("status") == "PASS" and summary.get("confirmation_n_cells") == 2048, "summary matches confirmed release")

    status = "PASS" if all(x["pass"] for x in findings) else "FAIL"
    report = {
        "version": "16.0.0",
        "status": status,
        "checks_passed": sum(bool(x["pass"]) for x in findings),
        "checks_total": len(findings),
        "findings": findings,
        "profile": {
            "unit_rows": len(unit), "unit_pareto_rows": len(ufront),
            "service_rows": len(service), "system_pareto_rows": len(sfront),
            "maximum_abs_charge_balance_error_pct": float(unit.charge_energy_balance_error_pct.abs().max()),
            "maximum_abs_discharge_balance_error_pct": float(unit.discharge_energy_balance_error_pct.abs().max()),
            "temperature_range_K": [float(unit.temperature_min_K.min()), float(unit.temperature_max_K.max())],
            "validation_holdout_RMSE_K": float(hold.RMSE_K),
            "validation_holdout_envelope_coverage_pct": float(hold.radial_envelope_coverage_pct),
        },
    }
    (R / "scientific_qa_report_v16.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
