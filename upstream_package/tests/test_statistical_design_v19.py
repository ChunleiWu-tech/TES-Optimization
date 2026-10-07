from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_v19"
sys.path.insert(0, str(ROOT / "scripts"))
from build_statistical_design_v19 import _ceil_array, _classification_changes_possible


class StatisticalDesignV19Tests(unittest.TestCase):
    def test_full_box_detects_interior_equality_missed_by_corners(self) -> None:
        a, b, epsilon = np.array([2.2]), np.array([3.2]), 1.6
        corners = [
            bool((_ceil_array(a + sa * epsilon, 1e-12)
                  == _ceil_array(b + sb * epsilon, 1e-12))[0])
            for sa in (-1, 1) for sb in (-1, 1)
        ]
        self.assertFalse(any(corners))
        self.assertTrue(bool(_classification_changes_possible(a, b, epsilon, 1e-12)[0]))

    def test_full_box_equal_pair_stability_requires_singleton_counts(self) -> None:
        a, b = np.array([2.4, 2.99]), np.array([2.6, 2.98])
        self.assertEqual(
            _classification_changes_possible(a, b, 0.02, 1e-12).tolist(), [False, True],
        )

    def test_registered_intervals_contain_point_estimates(self) -> None:
        table = pd.read_csv(RESULTS / "statistical_estimands_v19.csv")
        self.assertEqual(len(table), 7)
        self.assertTrue((table["confidence_low"] <= table["point_estimate"]).all())
        self.assertTrue((table["point_estimate"] <= table["confidence_high"]).all())
        self.assertTrue((table["n_clusters"] == 256).all())

    def test_scale_decomposition_telescopes_exactly(self) -> None:
        table = pd.read_csv(RESULTS / "scale_attenuation_decomposition_v19.csv")
        keys = ["group_id", "pair_id", "scenario_id", "base_design_id", "topology_id"]
        grouped = table.sort_values(keys + ["stage_order"]).groupby(keys, sort=False)
        summed = grouped["attenuation_log_magnitude"].sum().to_numpy(float)
        first = grouped["absolute_log_contrast_in"].first().to_numpy(float)
        last = grouped["absolute_log_contrast_out"].last().to_numpy(float)
        self.assertTrue(np.allclose(summed, first - last, atol=1e-12, rtol=0.0))

    def test_integer_boundary_is_stable_through_one_hundredth_module(self) -> None:
        table = pd.read_csv(RESULTS / "integer_boundary_stability_v19.csv")
        pooled = table[table["scenario_id"] == "ALL"]
        through = pooled[pooled["perturbation_module_count"] <= 0.01]
        self.assertTrue((through["classification_changes_possible"] == 0).all())
        at_two_hundredths = pooled[np.isclose(pooled["perturbation_module_count"], 0.02)]
        self.assertEqual(int(at_two_hundredths["classification_changes_possible"].iloc[0]), 9)

    def test_objective_resolution_membership_is_invariant(self) -> None:
        table = pd.read_csv(RESULTS / "pareto_release_stability_v19.csv")
        row = table[
            (table["front_definition"] == "objective_resolution_set")
            & (table["scenario_id"] == "ALL")
        ].iloc[0]
        self.assertEqual(int(row["v16_count"]), 52)
        self.assertEqual(int(row["v18_count"]), 52)
        self.assertEqual(float(row["jaccard_similarity"]), 1.0)


if __name__ == "__main__":
    unittest.main()
