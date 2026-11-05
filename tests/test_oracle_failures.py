"""Invalid arithmetic must never produce a positive grid-oracle verdict."""
import unittest
from unittest.mock import patch

import numpy as np

import bench_pendulum
import bounds
import cegis_c


class OracleFailureTests(unittest.TestCase):
    def setUp(self):
        self.theta = np.zeros((3, 3))

    def test_polynomial_rejects_nonfinite_bound_and_test_values(self):
        for bad in (float("nan"), float("inf"), -1.0):
            with self.subTest(bound=bad), patch.object(
                    bounds, "lipschitz_certified", return_value=bad):
                with self.assertRaisesRegex(ValueError, "invalid Lipschitz"):
                    cegis_c.verify(self.theta, 3)
        for bad in (float("nan"), float("inf")):
            with self.subTest(value=bad), patch.object(
                    bounds, "lipschitz_certified", return_value=1.0), patch.object(
                    cegis_c, "lam_max_C", return_value=np.full(9, bad)):
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    cegis_c.verify(self.theta, 3)

    def test_pendulum_rejects_nonfinite_bound_and_test_values(self):
        for bad in (float("nan"), float("inf"), -1.0):
            with self.subTest(bound=bad), patch.object(
                    bench_pendulum, "L_norm_p", return_value=bad):
                with self.assertRaisesRegex(ValueError, "invalid Lipschitz"):
                    bench_pendulum.verify_p(self.theta, 3)
        for bad in (float("nan"), float("inf")):
            with self.subTest(value=bad), patch.object(
                    bench_pendulum, "L_norm_p", return_value=1.0), patch.object(
                    bench_pendulum, "lam_max_C_p", return_value=np.full(9, bad)):
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    bench_pendulum.verify_p(self.theta, 3)


if __name__ == "__main__":
    unittest.main()
