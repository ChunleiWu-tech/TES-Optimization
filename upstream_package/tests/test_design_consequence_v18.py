from __future__ import annotations

import math
import unittest

from tespub.design_consequence_v18 import ceil_with_tolerance


class DesignConsequenceV18Tests(unittest.TestCase):
    def test_ceil_with_tolerance_preserves_exact_integer_boundary(self) -> None:
        self.assertEqual(ceil_with_tolerance(10.0, 1e-9), 10)
        self.assertEqual(ceil_with_tolerance(10.0 + 1e-12, 1e-9), 10)
        self.assertEqual(ceil_with_tolerance(10.0 + 1e-6, 1e-9), 11)

    def test_integer_boundary_margin_identity(self) -> None:
        for low, high in ((10.2, 10.8), (10.2, 11.1), (10.0, 10.1), (10.999, 10.9995)):
            with self.subTest(low=low, high=high):
                delta = high - low
                distance = ceil_with_tolerance(low, 1e-9) - low
                margin = delta - distance
                same_integer = ceil_with_tolerance(low, 1e-9) == ceil_with_tolerance(high, 1e-9)
                self.assertEqual(margin <= 1e-8, same_integer)
                self.assertTrue(math.isfinite(margin))


if __name__ == "__main__":
    unittest.main()

