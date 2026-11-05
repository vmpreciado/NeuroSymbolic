"""Independent mathematical checks for the revision's adaptive oracle and models."""
from fractions import Fraction
import unittest

import numpy as np

from revision_extensions import (ExactPoly, OracleBudget, adaptive_oracle,
    benchmark_dynamics, factor_metric, fit_surrogate, independent_pullback_identity,
    matrix_values, metric_and_contraction_nominal, polynomial_matrix_contraction,
    residual_constants, transfer_guarantee, bernstein_coefficients, split_bernstein,
    replay_oracle_certificate)


class AdaptiveExtensionTests(unittest.TestCase):
    def test_recenter_encloses_exact_values_after_cancellation(self):
        x = ExactPoly.coordinate(3, 0)
        y = ExactPoly.coordinate(3, 1)
        z = ExactPoly.coordinate(3, 2)
        p = (x - Fraction(7, 8)) ** 4 + (y + z) * Fraction(1, 10)
        lo, hi = [.5, -.25, .125], [1., .5, .25]
        bound = p.enclosure(lo, hi)
        for a in (Fraction(1, 2), Fraction(7, 8), Fraction(1)):
            for b in (Fraction(-1, 4), Fraction(1, 2)):
                for c in (Fraction(1, 8), Fraction(1, 4)):
                    exact = p.exact_value([a, b, c])
                    self.assertLessEqual(Fraction.from_float(float(bound.lo)), exact)
                    self.assertGreaterEqual(Fraction.from_float(float(bound.hi)), exact)

    def test_general_dimension_oracle_constant_matrix(self):
        n = 4
        matrix = [[ExactPoly.constant(n, -2 if i == j else .1) for j in range(n)] for i in range(n)]
        result = adaptive_oracle(matrix, [-1] * n, [1] * n, OracleBudget(target_margin=1))
        self.assertTrue(result['certified'])
        self.assertEqual(result['counters']['accepted_leaves'], 1)
        self.assertEqual(replay_oracle_certificate(result, matrix), 1)

    def test_bernstein_asymmetric_split_against_exact_values(self):
        import math
        x = ExactPoly.coordinate(1, 0)
        p = (x - Fraction(2, 3)) ** 5 - x ** 2 + Fraction(1, 7)
        lo, hi = .1, 2.4
        midpoint = lo + (hi - lo) / 2
        ratio = ((Fraction.from_float(midpoint) - Fraction.from_float(lo))
                 / (Fraction.from_float(hi) - Fraction.from_float(lo)))
        self.assertNotEqual(ratio, Fraction(1, 2))
        coefficients = bernstein_coefficients(p, [lo], [hi])
        left, right = split_bernstein(coefficients, 0, ratio)
        for a, b, c in ((lo, midpoint, left), (midpoint, hi, right)):
            degree = len(c.lo) - 1
            for t in (Fraction(0), Fraction(1, 7), Fraction(1, 2), Fraction(1)):
                point = Fraction.from_float(a) + t * (Fraction.from_float(b) - Fraction.from_float(a))
                weights = [math.comb(degree, k) * t ** k * (1 - t) ** (degree - k) for k in range(degree + 1)]
                lower = sum(Fraction.from_float(float(v)) * w for v, w in zip(c.lo, weights))
                upper = sum(Fraction.from_float(float(v)) * w for v, w in zip(c.hi, weights))
                self.assertLessEqual(lower, p.exact_value([point]))
                self.assertGreaterEqual(upper, p.exact_value([point]))

    def test_complete_cover_replay_and_missing_leaf_rejection(self):
        import copy
        x = ExactPoly.coordinate(1, 0)
        matrix = [[-(x - .5) ** 2 - .01]]
        result = adaptive_oracle(matrix, [.1], [2.4], OracleBudget(max_boxes=512, max_depth=16, target_margin=.001))
        self.assertTrue(result['certified'])
        self.assertGreater(replay_oracle_certificate(result, matrix), 1)
        broken = copy.deepcopy(result)
        broken['proof']['leaf_paths'].pop()
        with self.assertRaises(ValueError):
            replay_oracle_certificate(broken, matrix)

    def test_adjacent_float_endpoints_fail_closed_when_split_impossible(self):
        lo = .1
        hi = np.nextafter(lo, np.inf)
        x = ExactPoly.coordinate(1, 0)
        result = adaptive_oracle([[x - lo]], [lo], [hi], OracleBudget(max_boxes=10, target_margin=0))
        self.assertFalse(result['certified'])
        self.assertEqual(result['reason'], 'resolution_exhausted')

    def test_arithmetic_overflow_is_inconclusive(self):
        matrix = [[ExactPoly.constant(1, Fraction(10) ** 400)]]
        result = adaptive_oracle(matrix, [-1], [1], check_points=np.asarray([[0.]]))
        self.assertFalse(result['certified'])
        self.assertEqual(result['reason'], 'arithmetic_inconclusive')

    def test_indefinite_matrix_never_certifies(self):
        n = 3
        matrix = [[ExactPoly.constant(n, (-1, -1, .01)[i] if i == j else 0) for j in range(n)] for i in range(n)]
        result = adaptive_oracle(matrix, [-1] * n, [1] * n, OracleBudget(max_boxes=1))
        self.assertFalse(result['certified'])
        self.assertEqual(result['status'], 'inconclusive')

    def test_singular_nominal_preconditioner_cannot_certify(self):
        from unittest.mock import patch
        from rounding import Interval
        from revision_extensions import negative_definite_interval
        matrix = [[Interval(-2.1, .1), Interval(0.)],
                  [Interval(0.), Interval(-1.)]]
        with patch('revision_extensions.np.linalg.inv', return_value=np.zeros((2, 2))) as inverse:
            self.assertFalse(negative_definite_interval(matrix))
            inverse.assert_called_once()

    def test_budget_exhaustion_is_not_a_positive_answer(self):
        x = ExactPoly.coordinate(2, 0)
        z = ExactPoly.constant(2, 0)
        # Negative everywhere, but the wide coefficient enclosure is inconclusive.
        matrix = [[-(x - .5) ** 2 - .01, z], [z, ExactPoly.constant(2, -1)]]
        result = adaptive_oracle(matrix, [-1, -1], [1, 1], OracleBudget(max_boxes=1, max_depth=4))
        self.assertFalse(result['certified'])
        self.assertEqual(result['reason'], 'box_budget_exhausted')

    def test_coupled_diffeomorphism_identity(self):
        result = independent_pullback_identity()
        self.assertTrue(result['conjugacy_identity_exact'])
        self.assertTrue(result['contraction_identity_exact'])
        f = benchmark_dynamics(3)
        self.assertTrue(f[2].derivative(0).coeffs)
        self.assertTrue(f[2].derivative(1).coeffs)

    def test_nominal_learner_matches_independent_exact_polynomials(self):
        rng = np.random.RandomState(13)
        theta = rng.normal(0, .4, (6, 4))
        points = rng.uniform(-1, 1, (17, 3))
        dynamics = benchmark_dynamics(3)
        metric = factor_metric(theta, .1)
        C = polynomial_matrix_contraction(metric, dynamics, .4)
        nominal_M, nominal_C = metric_and_contraction_nominal(theta, points, dynamics, .1, .4)
        np.testing.assert_allclose(nominal_M, matrix_values(metric, points), rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(nominal_C, matrix_values(C, points), rtol=3e-13, atol=3e-13)

    def test_residual_bounds_cover_known_polynomial_errors(self):
        truth = benchmark_dynamics(2)
        x = ExactPoly.coordinate(2, 0)
        fit = [truth[0] + Fraction(1, 1000) * x, truth[1]]
        bound = residual_constants(truth, fit, B=2., depth=1)
        self.assertGreaterEqual(bound['delta0_upper'], .002)
        self.assertGreaterEqual(bound['delta1_upper'], .001)
        self.assertLess(bound['delta0_upper'], .002000000001)

    def test_fitted_model_and_transfer_do_not_assume_exact_truth(self):
        fit, config = fit_surrogate()
        bound = residual_constants(benchmark_dynamics(2), fit, 2., depth=1)
        self.assertTrue(config['truth_not_used_in_regression'])
        self.assertGreater(bound['delta1_upper'], 0)
        result = transfer_guarantee(.01, 5., 2., .0001, .0001, .4, .1)
        self.assertTrue(result['certified'])
        self.assertLessEqual(result['remaining_margin_lower'], .0088)
        bad = transfer_guarantee(.001, 5., 2., 1., 1., .4, .1)
        self.assertFalse(bad['certified'])
        self.assertIsNone(bad['true_rate_lower'])


if __name__ == '__main__':
    unittest.main()
