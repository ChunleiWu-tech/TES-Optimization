from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from tespub import __version__
from tespub.codesign_v16 import _array_header_hydraulics, generate_geometry_pool, load_config, size_fixed_service
from tespub.fluid_properties import fluid_properties, get_model_spec
from tespub.ltne_v16 import BedGeometryV16, evaluate_packed_bed_v16


class V16ScientificTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(ROOT / "config" / "study_v16.json")
        cls.results = ROOT / "results_v16"

    def test_version_single_source(self) -> None:
        self.assertEqual(__version__, "16.0.0")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('version = "16.0.0"', pyproject)

    def test_property_equation_and_domain_gate(self) -> None:
        cp = fluid_properties("SOLAR_SALT_60_40_V10", 525.0 + 273.15).cp_J_kgK
        self.assertAlmostEqual(cp, 1443.0 + 0.172 * 525.0, places=8)
        with self.assertRaises(ValueError):
            fluid_properties("AIP_FLINAK_2017_V10", 499.0 + 273.15)

    def test_registered_models_cover_common_duty(self) -> None:
        lo, hi = self.cfg["common_duty_C"]
        for model_id in self.cfg["fluid_model_ids"]:
            spec = get_model_spec(model_id)
            self.assertLessEqual(spec.valid_temperature_min_C, lo)
            self.assertGreaterEqual(spec.valid_temperature_max_C, hi)

    def test_lhs_and_topology_expansion_are_deterministic(self) -> None:
        a = generate_geometry_pool(self.cfg, 24, 17)
        b = generate_geometry_pool(self.cfg, 24, 17)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), 24 * 3)
        self.assertEqual(set(a["gradient_orientation"]), {"none", "fine_hot", "coarse_hot"})
        for field, bounds in self.cfg["design_space"].items():
            self.assertTrue(a[field].between(*bounds).all())

    def test_conservative_solver_has_no_clipping_or_temperature_overshoot(self) -> None:
        geom = BedGeometryV16(0.90, 1.50, 0.50, 0.012, "graded_two_layer", 1.8, "fine_hot", 0.5)
        cp = fluid_properties("SOLAR_SALT_60_40_V10", 798.15).cp_J_kgK
        flow = 18_000.0 / (cp * 50.0)
        r = evaluate_packed_bed_v16(
            model_id="SOLAR_SALT_60_40_V10", design_id="TEST", geometry=geom,
            mass_flow_kg_s=flow, cold_temperature_K=773.15, hot_temperature_K=823.15,
            solid_density_kg_m3=2732.0, solid_cp_J_kgK=1295.0,
            solid_thermal_conductivity_W_mK=0.592, n_cells=32,
        )
        self.assertFalse(r["temperature_clipping_used"])
        self.assertGreaterEqual(r["temperature_min_K"], 773.15 - 1e-8)
        self.assertLessEqual(r["temperature_max_K"], 823.15 + 1e-8)
        self.assertLess(abs(r["charge_energy_balance_error_pct"]), 1e-6)
        self.assertLess(abs(r["discharge_energy_balance_error_pct"]), 1e-6)
        self.assertTrue(r["counterflow_discharge"])
        self.assertTrue(r["cycle_coupled_discharge"])
        self.assertGreater(r["charge_final_inventory_fraction"], 0.99)
        self.assertTrue(r["threshold_crossing_interpolation_used"])
        self.assertLessEqual(r["time_step_s"], 5.0 + 1e-9)
        self.assertEqual(r["temperature_extrema_scope"], "initial state and every explicit transient time step")
        self.assertEqual(r["thermal_front_crossings_90"], 1)
        self.assertEqual(r["thermal_front_crossings_10"], 1)
        self.assertEqual(r["thermal_front_nonmonotonic_steps"], 0)

    def test_nonzero_wall_loss_requires_an_explicit_sink_temperature(self) -> None:
        geom = BedGeometryV16(0.90, 1.50, 0.50, 0.012)
        cp = fluid_properties("SOLAR_SALT_60_40_V10", 798.15).cp_J_kgK
        args = dict(
            model_id="SOLAR_SALT_60_40_V10", design_id="LOSS_BOUNDARY", geometry=geom,
            mass_flow_kg_s=18_000.0 / (cp * 50.0), cold_temperature_K=773.15,
            hot_temperature_K=823.15, solid_density_kg_m3=2732.0,
            solid_cp_J_kgK=1295.0, solid_thermal_conductivity_W_mK=0.592,
            n_cells=32, wall_loss_W_m3K=0.2,
        )
        with self.assertRaises(ValueError):
            evaluate_packed_bed_v16(**args)
        solved = evaluate_packed_bed_v16(ambient_temperature_K=298.15, **args)
        self.assertAlmostEqual(solved["ambient_temperature_C"], 25.0, places=10)

    def test_threshold_and_transport_temperature_controls_are_explicit(self) -> None:
        geom = BedGeometryV16(0.90, 1.50, 0.50, 0.012)
        cp = fluid_properties("SOLAR_SALT_60_40_V10", 798.15).cp_J_kgK
        common = dict(
            model_id="SOLAR_SALT_60_40_V10", geometry=geom,
            mass_flow_kg_s=18_000.0 / (cp * 50.0),
            cold_temperature_K=773.15, hot_temperature_K=823.15,
            solid_density_kg_m3=2732.0, solid_cp_J_kgK=1295.0,
            solid_thermal_conductivity_W_mK=0.592, n_cells=32,
        )
        low = evaluate_packed_bed_v16(
            design_id="G085_COLD", minimum_outlet_grade_fraction=0.85,
            transport_property_temperature_K=773.15, **common,
        )
        high = evaluate_packed_bed_v16(
            design_id="G095_HOT", minimum_outlet_grade_fraction=0.95,
            transport_property_temperature_K=823.15, **common,
        )
        self.assertGreater(low["deliverable_energy_MWh"], high["deliverable_energy_MWh"])
        self.assertAlmostEqual(low["transport_property_temperature_C"], 500.0, places=8)
        self.assertAlmostEqual(high["transport_property_temperature_C"], 550.0, places=8)
        self.assertIn("within-step interpolation", low["deliverable_energy_definition"])

    def test_transport_endpoint_service_power_remains_exact(self) -> None:
        unit = pd.read_csv(self.results / "codesign_results_v16.csv").iloc[[0]].copy()
        local = json.loads(json.dumps(self.cfg))
        local["service_scenarios"] = [dict(local["service_scenarios"][0])]
        local["service_scenarios"][0]["transport_property_temperature_K"] = 773.15
        result = size_fixed_service(unit, local).iloc[0]
        multiplier = float(unit.iloc[0].get("cp_multiplier", 1.0))
        endpoint_cp = fluid_properties(str(result["fluid_model_id"]), 773.15).cp_J_kgK * multiplier
        resolved_power_W = (
            float(result["parallel_unit_count"])
            * float(result["resolved_unit_mass_flow_kg_s"])
            * endpoint_cp
            * 50.0
        )
        self.assertAlmostEqual(
            resolved_power_W,
            float(result["target_thermal_power_MW"]) * 1e6,
            places=5,
        )

    def test_graded_topology_is_physically_active(self) -> None:
        cp = fluid_properties("SOLAR_SALT_60_40_V10", 798.15).cp_J_kgK
        flow = 18_000.0 / (cp * 50.0)
        common = dict(
            model_id="SOLAR_SALT_60_40_V10", mass_flow_kg_s=flow,
            cold_temperature_K=773.15, hot_temperature_K=823.15,
            solid_density_kg_m3=2732.0, solid_cp_J_kgK=1295.0,
            solid_thermal_conductivity_W_mK=0.592, n_cells=32,
        )
        h = evaluate_packed_bed_v16(design_id="H", geometry=BedGeometryV16(0.9,1.5,0.5,0.012), **common)
        f = evaluate_packed_bed_v16(design_id="F", geometry=BedGeometryV16(0.9,1.5,0.5,0.012,"graded_two_layer",1.8,"fine_hot",0.5), **common)
        c = evaluate_packed_bed_v16(design_id="C", geometry=BedGeometryV16(0.9,1.5,0.5,0.012,"graded_two_layer",1.8,"coarse_hot",0.5), **common)
        self.assertGreater(f["pressure_drop_kPa"], h["pressure_drop_kPa"])
        self.assertGreater(c["pressure_drop_kPa"], h["pressure_drop_kPa"])
        self.assertNotAlmostEqual(f["thermal_front_thickness_ratio"], c["thermal_front_thickness_ratio"], places=5)

    def test_all_designs_are_confirmed_at_registered_release_resolution(self) -> None:
        d = pd.read_csv(self.results / "codesign_results_v16.csv")
        sc = pd.read_csv(self.results / "screen_to_confirmation_v16.csv")
        self.assertEqual(len(d), 3840)
        self.assertTrue(d["design_id"].is_unique)
        self.assertEqual(set(d["n_cells"]), {int(self.cfg["sampling"]["confirmation_n_cells"])})
        self.assertEqual(len(sc), 3840)
        self.assertLess(sc["charge_t90_h_relative_difference_pct"].abs().quantile(0.95), 2.0)
        self.assertLess(sc["high_grade_discharge_fraction_absolute_difference"].abs().quantile(0.95), 0.08)

    def test_pair_gate_blocks_deployment_overclaim(self) -> None:
        gate = pd.read_csv(ROOT / "data" / "salt_skeleton_pair_registry.csv")
        self.assertTrue(gate["physics_comparison_eligible"].astype(str).str.lower().eq("true").all())
        self.assertTrue(gate["engineering_deployment_eligible"].astype(str).str.lower().eq("false").all())
        front = pd.read_csv(self.results / "system_pareto_front_v16.csv")
        self.assertFalse(front["engineering_deployment_eligible"].astype(bool).any())

    def test_particle_hierarchy_supports_effective_resistance(self) -> None:
        p = pd.read_csv(self.results / "particle_model_hierarchy_v16.csv")
        self.assertLess(p["effective_resistance_error_vs_radial_pct"].abs().max(), 3.0)
        self.assertGreater(p["lumped_error_vs_radial_pct"].abs().max(), 50.0)

    def test_external_validation_is_holdout_and_claim_bounded(self) -> None:
        m = pd.read_csv(self.results / "external_validation_metrics_v16.csv")
        self.assertEqual(set(m["role"]), {"calibration", "validation", "diagnostic"})
        cal = m.loc[m.role == "calibration"].iloc[0]
        val = m.loc[m.role == "validation"].iloc[0]
        self.assertGreater(val.RMSE_K, cal.RMSE_K)
        self.assertIn("cooling held out", val.claim_boundary.lower())
        diag = m.loc[m.role == "diagnostic"].iloc[0]
        self.assertIn("figure-digitized", diag.claim_boundary.lower())

    def test_system_header_is_geometry_normalized_and_recomputable(self) -> None:
        s = pd.read_csv(self.results / "fixed_service_results_v16.csv")
        q = s[(s.scenario_id == "S10MW_8H") & (s.base_design_id == "G0001") & (s.topology_id.str.startswith("T1"))]
        self.assertEqual(len(q), 5)
        unit = pd.read_csv(self.results / "codesign_results_v16.csv").set_index("design_id")
        T = 0.5 * sum(x + 273.15 for x in self.cfg["common_duty_C"])
        for _, r in q.iterrows():
            fluid = fluid_properties(r.fluid_model_id, T)
            design = unit.loc[r.design_id]
            rho = fluid.density_kg_m3 * float(design.get("density_multiplier", 1.0))
            mu = fluid.viscosity_Pa_s * float(design.get("viscosity_multiplier", 1.0))
            expected = _array_header_hydraulics(
                parallel_unit_count=int(r.parallel_unit_count),
                total_volumetric_flow_m3_s=float(r.parallel_unit_count) * float(r.resolved_unit_mass_flow_kg_s) / rho,
                density_kg_m3=rho,
                viscosity_Pa_s=mu,
                module_diameter_m=float(design.internal_diameter_m),
                config=self.cfg,
            )
            self.assertAlmostEqual(r.header_pressure_drop_kPa, float(expected["header_pressure_drop_Pa"]) / 1000.0, places=10)
            self.assertGreaterEqual(int(r.header_array_rows) * int(r.header_array_max_columns), int(r.parallel_unit_count))
            self.assertIn("Darcy-Weisbach", r.header_hydraulic_model)
            self.assertIn("frictional energy", r.pump_energy_definition)

    def test_service_is_closed_at_resolved_flow(self) -> None:
        s = pd.read_csv(self.results / "fixed_service_results_v16.csv")
        self.assertGreaterEqual(s["service_capacity_closure_error_pct"].min(), -1e-8)
        self.assertFalse(s["adjacent_lower_count_feasible"].astype(bool).any())
        expected = s["parallel_unit_count"] * s["resolved_unit_deliverable_energy_MWh"]
        np.testing.assert_allclose(expected, s["installed_high_grade_capacity_MWh"], rtol=0.0, atol=1e-10)
        self.assertTrue((s["minimum_outlet_grade_fraction"] == 0.9).all())
        self.assertTrue(s["deliverable_energy_definition"].str.contains("time integral", case=False).all())
        self.assertTrue(s["threshold_crossing_interpolation_used"].astype(bool).all())
        self.assertTrue(np.isfinite(s["charge_discharge_t90_asymmetry_pct"]).all())

    def test_retained_decisions_pass_the_4096_cell_release_gate(self) -> None:
        g = pd.read_csv(self.results / "decision_grid_refinement_v16.csv")
        self.assertGreater(len(g), 0)
        self.assertEqual(set(g["production_n_cells"]), {2048})
        self.assertEqual(set(g["refinement_n_cells"]), {4096})
        self.assertTrue(g["grid_refinement_accepted"].astype(bool).all())
        self.assertLessEqual(g["deliverable_energy_change_pct"].abs().max(), 0.5)
        self.assertLessEqual(g["high_grade_fraction_change"].abs().max(), 0.005)

    def test_decision_precision_and_uncertainty_cover_both_services(self) -> None:
        precision = pd.read_csv(self.results / "pareto_precision_sensitivity_v16.csv")
        self.assertEqual(set(precision["scenario_id"]), {"S10MW_8H", "S50MW_6H"})
        self.assertEqual(set(precision["decision_precision_multiplier"]), {0.0, 0.5, 1.0, 2.0})
        robust = pd.read_csv(self.results / "bounded_robustness_v16.csv")
        self.assertEqual(set(robust["scenario_id"]), {"S10MW_8H", "S50MW_6H"})
        assessed = robust[robust["n_stress_samples"] > 0]
        self.assertTrue((assessed["n_stress_samples"] == 32).all())

    def test_standby_loss_is_monotone(self) -> None:
        d = pd.read_csv(self.results / "standby_loss_v16.csv")
        for _, part in d.groupby(["design_id", "wall_loss_W_m3K"]):
            ordered = part.sort_values("standby_duration_h")
            self.assertTrue(np.all(np.diff(ordered["standby_sensible_loss_pct"]) >= -1e-10))

    def test_model_stress_is_ordered_and_geometry_scaled(self) -> None:
        d = pd.read_csv(self.results / "model_form_stress_v16.csv")
        self.assertEqual(set(d["stress_id"]), {"intrinsic_adiabatic", "bounded_moderate", "bounded_severe"})
        feasible = d.groupby("stress_id")["service_admissible"].sum()
        self.assertGreaterEqual(feasible["intrinsic_adiabatic"], feasible["bounded_moderate"])
        self.assertGreaterEqual(feasible["bounded_moderate"], feasible["bounded_severe"])
        nonzero = d[d["surface_heat_transfer_coefficient_W_m2K"] > 0]
        self.assertTrue((nonzero["wall_loss_W_m3K"] > 0).all())
        self.assertTrue(np.allclose(
            nonzero["stress_ambient_temperature_C"],
            float(self.cfg["model_form_stress"]["ambient_temperature_C"]),
        ))
        self.assertTrue((d["stress_n_cells"] == int(self.cfg["model_form_stress"]["n_cells"])).all())

    def test_flow_re_solution_never_increases_energy_limited_integer(self) -> None:
        s = pd.read_csv(self.results / "fixed_service_results_v16.csv")
        delta = s["resolved_energy_limited_unit_count"] - s["nominal_energy_limited_unit_count"]
        self.assertTrue((delta <= 0).all())
        self.assertGreater((delta < 0).sum(), 0)
        self.assertGreater((delta == 0).sum(), 0)

    def test_outputs_and_fronts_are_complete(self) -> None:
        summary = json.loads((self.results / "analysis_summary_v16.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["n_unit_cases"], 3840)
        self.assertEqual(summary["n_fixed_service_cases"], 7680)
        self.assertGreater(summary["n_unit_pareto"], 0)
        self.assertGreater(summary["n_system_pareto"], 0)
        self.assertEqual(summary["service_closure"]["adjacent_lower_feasible_cases"], 0)
        self.assertEqual(len(summary["outlet_grade_threshold_sensitivity"]), 6)
        self.assertGreater(summary["service_monotonicity_audit"]["integer_counts_scanned"], 0)
        self.assertGreater(summary["service_monotonicity_audit"]["stratified_probe_count_evaluations"], 0)
        self.assertTrue(summary["numerical_decision_validation"]["all_cases_accepted"])
        self.assertEqual(
            summary["service_monotonicity_audit"]["cases_matching_reported_minimum"],
            summary["service_monotonicity_audit"]["retained_decision_cases"],
        )
        for name in (
            "outlet_grade_threshold_sensitivity_v16.csv",
            "module_scale_sensitivity_v16.csv",
            "transport_temperature_sensitivity_v16.csv",
            "service_monotonicity_audit_v16.csv",
            "service_monotonicity_probe_v16.csv",
            "decision_grid_refinement_v16.csv",
        ):
            self.assertTrue((self.results / name).is_file())
        scale = pd.read_csv(self.results / "module_scale_sensitivity_v16.csv")
        self.assertTrue((scale["scaled_n_cells"] == (
            int(self.cfg["sampling"]["confirmation_n_cells"]) * scale["linear_scale_factor"]
        )).all())
        self.assertIn("corrosion", summary["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
