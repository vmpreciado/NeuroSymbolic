"""Independent spectral checks for the published-objective adaptation."""
import unittest

import autograd.numpy as np
from autograd import grad
import numpy as onp

from application_baseline import pdi_loss_from_entries, protocol_description


class ApplicationBaselineTests(unittest.TestCase):
    def test_matches_independent_symmetric_eigensolver(self):
        rng = onp.random.default_rng(473)
        matrices = rng.normal(size=(41, 2, 2))
        matrices = (matrices + matrices.transpose(0, 2, 1)) / 2
        margin = 0.01
        eigs = onp.linalg.eigvalsh(matrices + margin * onp.eye(2))
        expected = onp.mean(onp.maximum(eigs, 0.0) ** 2)
        observed = pdi_loss_from_entries(
            matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 1, 1], margin)
        self.assertAlmostEqual(float(observed), float(expected), places=13)

    def test_counts_both_positive_eigenvalues(self):
        # Both 2 and 3 contribute; using only lambda_max would give 4.5.
        self.assertEqual(float(pdi_loss_from_entries(np.array([2.0]),
            np.array([0.0]), np.array([3.0]), 0.0)), 6.5)

    def test_zero_set_matches_raw_matrix_margin(self):
        margin = 0.01
        for eigenvalues, zero in [((-1.0, -0.02), True),
                                  ((-1.0, 0.0), False),
                                  ((-1.0, 0.5), False)]:
            rotation = onp.array([[0.6, -0.8], [0.8, 0.6]])
            matrix = rotation @ onp.diag(eigenvalues) @ rotation.T
            loss = float(pdi_loss_from_entries(np.array([matrix[0, 0]]),
                np.array([matrix[0, 1]]), np.array([matrix[1, 1]]), margin))
            self.assertEqual(loss == 0.0, zero)

    def test_gradient_agrees_with_independent_finite_difference(self):
        theta = onp.array([0.4, -0.3, 1.2])
        objective = lambda p: pdi_loss_from_entries(p[0:1], p[1:2], p[2:3], 0.01)
        observed = grad(objective)(theta)
        def reference(p):
            matrix = onp.array([[p[0], p[1]], [p[1], p[2]]])
            return onp.mean(onp.maximum(onp.linalg.eigvalsh(matrix)+0.01, 0.0)**2)
        expected = []
        for k in range(3):
            delta = onp.zeros(3); delta[k] = 1e-6
            expected.append((reference(theta+delta)-reference(theta-delta))/(2e-6))
        onp.testing.assert_allclose(observed, expected, rtol=1e-7, atol=1e-8)

    def test_repeated_eigenvalue_gradient_finite(self):
        objective = lambda p: pdi_loss_from_entries(p[0:1], p[1:2], p[2:3], 0.01)
        derivative = grad(objective)(np.array([1.0, 0.0, 1.0]))
        onp.testing.assert_allclose(derivative, [1.01, 0.0, 1.01], rtol=1e-14)

    def test_provenance_does_not_claim_full_method_reproduction(self):
        self.assertIn("controlled adaptation", protocol_description()["name"])
        self.assertIn("not original full-method reproduction", protocol_description()["scope"])


if __name__ == "__main__":
    unittest.main()
