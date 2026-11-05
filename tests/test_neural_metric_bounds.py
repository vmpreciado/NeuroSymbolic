"""Independent checks of global neural metric bounds and exported guarantees."""
from fractions import Fraction
import math
import unittest
from unittest.mock import patch

import gmpy2 as gp
import numpy as np

import bench_neural as neural


def mp(value):
    return gp.mpfr(float(value))


def reference_metric(parameters, state):
    """High-precision metric assembled directly from the stored weights."""
    W1, b1, W2, b2 = neural._unpack(parameters)
    x, y = map(mp, state)
    hidden = [gp.tanh(mp(row[0])*x+mp(row[1])*y+mp(bias))
              for row, bias in zip(W1, b1)]
    ell = [sum((mp(weight)*value for weight, value in zip(row, hidden)), mp(bias))
           for row, bias in zip(W2, b2)]
    u, v, w = ell
    return mp(neural.ALPHA)+u*u+v*v, v*w, mp(neural.ALPHA)+w*w


class NeuralMetricBoundTests(unittest.TestCase):
    def test_global_bound_contains_independent_high_precision_eigenvalues(self):
        rng = np.random.default_rng(197)
        points = np.vstack((rng.uniform(-neural.BOX, neural.BOX, (19, 2)),
                            [[-2.,-2.],[-2.,2.],[2.,-2.],[2.,2.],[0.,0.]]))
        with gp.context(precision=320):
            for seed in (0, 2, 7):
                parameters = neural.init_params(seed)
                bounds = neural.metric_bounds_n(parameters, depth=3)
                for point in points:
                    p, q, r = reference_metric(parameters, point)
                    radius = gp.sqrt(((p-r)/2)**2+q*q)
                    self.assertGreaterEqual((p+r)/2-radius, mp(bounds['a']))
                    self.assertLessEqual((p+r)/2+radius, mp(bounds['b']))
                self.assertTrue(bounds['metric_bounds_outward_rounded'])

    def test_constant_network_frobenius_bound_is_outward(self):
        parameters = np.zeros(6*neural.H_UNITS+3)
        parameters[-3:] = [2., -1., 3.]
        bounds = neural.metric_bounds_n(parameters, depth=0)
        with gp.context(precision=320):
            p, q, r = reference_metric(parameters, [0., 0.])
            exact_frobenius = gp.sqrt(p*p+2*q*q+r*r)
            self.assertGreaterEqual(mp(bounds['b']), exact_frobenius)
            self.assertLess(mp(bounds['b'])-exact_frobenius, gp.mpfr('1e-12'))

    def test_quantitative_outputs_have_safe_rounding_directions(self):
        parameters = neural.init_params(4)
        for normalized in (False, True):
            for slack in (0.0, 0.03, np.nextafter(0., 1.)):
                out = neural.certified_metric_outputs_n(parameters, slack, normalized, depth=2)
                a, b, s = map(Fraction.from_float, (out['a'], out['b'], slack))
                eta = Fraction.from_float(out['raw_margin_lower'])
                expected_eta = 2*a*s if normalized else s
                self.assertTrue(0 <= eta <= expected_eta)
                rate = Fraction.from_float(out['rate_lower'])
                self.assertTrue(Fraction.from_float(neural.RATE) <= rate <=
                                Fraction.from_float(neural.RATE)+eta/(2*b))
                overshoot = Fraction.from_float(out['overshoot_upper'])
                self.assertGreaterEqual(overshoot*overshoot, b/a)

    def test_invalid_metric_and_slack_fail_closed(self):
        parameters = neural.init_params(0)
        parameters[0] = math.nan
        with self.assertRaises(ArithmeticError):
            neural.metric_bounds_n(parameters)
        for slack in (-0.1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                neural.certified_metric_outputs_n(neural.init_params(0), slack)
        for depth in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                neural.metric_bounds_n(neural.init_params(0), depth)

    def test_positive_verifier_cannot_omit_failed_metric_bounds(self):
        proof = {'complete': True, 'fallback': False, 'certified': True,
                 'worst': -1., 'viol': np.zeros((0, 2)),
                 'boxes_evaluated': 1, 'gridpoints_covered': 9}
        with patch.object(neural, 'L_norm_n', return_value=0.), \
             patch.object(neural, 'lam_max_C_n', side_effect=lambda p, x, **kw: -np.ones(len(x))), \
             patch.object(neural, '_rounded_grid_boxes', return_value=proof), \
             patch.object(neural, 'metric_bounds_n', side_effect=ArithmeticError('invalid bounds')):
            with self.assertRaises(ArithmeticError):
                neural.verify_n(neural.init_params(0), 3)


if __name__ == '__main__':
    unittest.main()
