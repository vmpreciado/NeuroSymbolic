"""Checks for certification configuration, streaming selection, and numeric inputs."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import bench_neural as neural
import fill_tex
import smoke_p1


class NeuralStreamingTests(unittest.TestCase):
    def test_streaming_matches_full_grid_verdict_worst_and_counterexamples(self):
        n = 17
        p = neural.init_params(3)
        X, s = neural.grid_n(n)
        for normalized in (True, False):
            L = neural.L_norm_n(p) if normalized else neural.L_raw_n(p)
            tau = (s[1] - s[0]) * math.sqrt(2) / 2
            g = np.asarray(neural.lam_max_C_n(p, X, normalized=normalized))
            idx = np.flatnonzero(g > -L * tau)
            idx = idx[np.argsort(-g[idx], kind="stable")[:7]]
            with patch.object(neural, "VERIFY_BLOCK_POINTS", 51):
                actual = neural.verify_n(p, n, normalized=normalized, k_worst=7)
            self.assertEqual(actual["certified"], len(idx) == 0)
            self.assertAlmostEqual(actual["worst"], float(g.max()), places=12)
            np.testing.assert_array_equal(actual["viol"], X[idx])

    def test_exact_ties_preserve_grid_order_across_chunks(self):
        X, _ = neural.grid_n(9)
        with patch.object(neural, "VERIFY_BLOCK_POINTS", 18), \
                patch.object(neural, "L_norm_n", return_value=1.0), \
                patch.object(neural, "lam_max_C_n", side_effect=lambda p, x, **kw: np.ones(len(x))):
            actual = neural.verify_n(neural.init_params(0), 9, k_worst=5)
        self.assertFalse(actual["certified"])
        np.testing.assert_array_equal(actual["viol"], X[:5])

    def test_nonfinite_values_cannot_certify(self):
        with patch.object(neural, "L_norm_n", return_value=1.0), \
                patch.object(neural, "lam_max_C_n", side_effect=lambda p, x, **kw: np.full(len(x), np.nan)):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                neural.verify_n(neural.init_params(0), 3)


class ReproducibilityInputTests(unittest.TestCase):
    def test_smoke_uses_recorded_grid_depth_and_iteration_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "traces.json").write_text(json.dumps({"config": {
                "grid_operating": 321, "depth": 2, "max_iter": 9}}))
            (root / "gridcost.json").write_text(json.dumps({"config": {
                "grid_operating": 321, "depth": 2}, "ratios": [{"seed": 4}]}))
            with patch.object(smoke_p1, "RESULTS", folder):
                self.assertEqual(smoke_p1._source_configuration(), (321, 2, 9, [4]))
                gc = json.loads((root / "gridcost.json").read_text())
                gc["config"]["grid_operating"] = 1001
                (root / "gridcost.json").write_text(json.dumps(gc))
                with self.assertRaisesRegex(ValueError, "disagree"):
                    smoke_p1._source_configuration()

    def test_missing_results_source_fails(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(fill_tex, "RESULTS", folder):
            with self.assertRaisesRegex(ValueError, "required results source"):
                fill_tex._load("traces.json")

    def test_missing_and_nonfinite_manuscript_numbers_fail(self):
        for value in (None, float("nan"), float("inf"), "19"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fill_tex.fmt(value)


if __name__ == "__main__":
    unittest.main()
