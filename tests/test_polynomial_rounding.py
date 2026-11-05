"""Independent exact-coefficient and high-precision polynomial oracle checks."""
from fractions import Fraction
import unittest
from unittest.mock import patch

import gmpy2
import numpy as np

import bounds
import cegis_c
import residual
from rounding import float_down, float_up


class ExactPolynomialTests(unittest.TestCase):
    def test_assembly_tracks_rounding_before_coefficients_are_formed(self):
        theta = np.array([[1/3, 1e8, 2**-30], [.1, -.3, .7], [.9, .2, -.4]])
        exact, nominal = bounds.assemble_exact(theta), bounds.assemble(theta)
        differences = []
        for name in ('m11', 'm12', 'm22', 'c11', 'c12', 'c22'):
            for key, value in exact[name].coeffs.items():
                self.assertIsInstance(value, Fraction)
                differences.append(value != Fraction.from_float(nominal[name].coeffs.get(key, 0.0)))
        self.assertTrue(any(differences), 'test must expose a pre-enclosure coefficient rounding error')

    def test_exact_shift_identity_and_outward_subdivision_bounds(self):
        p = bounds.ExactPoly2({(3, 0): 1/3, (1, 2): -.7,
                               (0, 0): 1e-20, (1, 0): 1e16})
        c1, c2 = Fraction(1, 3), Fraction(-2, 7)
        q = p.shift(c1, c2)
        for x, y in [(Fraction(0), Fraction(0)), (Fraction(1, 17), Fraction(-1, 19))]:
            self.assertEqual(q.eval(x, y), p.eval(c1+x, c2+y))
        B = 1.3
        upper = Fraction.from_float(p.supabs_subdiv(B, 2))
        lower = Fraction.from_float(p.inf_subdiv(B, 2))
        for x in np.linspace(-B, B, 13):
            for y in np.linspace(-B, B, 13):
                value = p.eval(x, y)
                self.assertLessEqual(abs(value), upper)
                self.assertLessEqual(lower, value)

    def test_interval_horner_encloses_exact_rational_values(self):
        p = bounds.ExactPoly2({(0, 0): Fraction(1, 3), (1, 0): 1e16,
                               (2, 0): -1e16, (0, 2): -2**-1022,
                               (2, 1): -.1})
        X = np.array([[0., 0.], [1., 0.], [np.nextafter(1., 2.), -2.],
                      [2**-1022, 2**-1074], [-2., 2.], [.7, -.9]])
        enclosure = p.interval_eval(X)
        for i, point in enumerate(X):
            exact = p.eval(*point)
            self.assertLessEqual(Fraction.from_float(float(enclosure.lo[i])), exact)
            self.assertGreaterEqual(Fraction.from_float(float(enclosure.hi[i])), exact)

    def test_contraction_and_normalized_quotient_enclose_high_precision(self):
        rng = np.random.default_rng(14)
        X = np.array([[-2., -2.], [2., 2.], [0., 0.], [.1, -.3],
                      [np.nextafter(2., 0.), 2**-1022]])
        for _ in range(4):
            theta = rng.uniform(-1, 1, (3, 3))
            P = bounds.assemble_exact(theta)
            for normalized in (False, True):
                enclosure = bounds.contraction_grid_interval(theta, X, normalized, P=P)
                for i, point in enumerate(X):
                    vals = [P[k].eval(*point) for k in ('c11', 'c12', 'c22', 'v')]
                    with gmpy2.context(precision=512):
                        a, b, c, v = [gmpy2.mpfr(gmpy2.mpq(x.numerator, x.denominator)) for x in vals]
                        middle, difference = (a+c)/2, (a-c)/2
                        reference = middle + gmpy2.sqrt(difference*difference+b*b)
                        if normalized:
                            reference /= v
                        self.assertLessEqual(gmpy2.mpfr(float(enclosure.lo[i])), reference)
                        self.assertGreaterEqual(gmpy2.mpfr(float(enclosure.hi[i])), reference)


class GridCertificateTests(unittest.TestCase):
    def test_nominal_screen_cannot_create_false_certificate(self):
        theta = np.zeros((3, 3))
        theta[0, 0] = theta[2, 0] = 1.
        # Force a fabricated nominal pass for an actually violating identity metric.
        with patch.object(cegis_c, 'lam_max_C_norm', return_value=np.full(9, -100.)):
            verdict = cegis_c.verify(theta, 3, normalized=True)
        self.assertTrue(verdict['grid_enclosed'])
        self.assertFalse(verdict['certified'])
        self.assertGreater(verdict['worst'], 0)

    def test_positive_answer_has_outward_grid_and_lower_uniform_slack(self):
        # The global Lipschitz bound remains valid on this small test domain.
        with patch.object(cegis_c, 'BOX', .25):
            verdict = cegis_c.verify(np.zeros((3, 3)), 9, normalized=True)
        self.assertTrue(verdict['certified'])
        self.assertTrue(verdict['grid_enclosed'])
        self.assertTrue(verdict['outward_rounded'])
        self.assertTrue(verdict['slack_is_certified'])
        exact_expression = (-Fraction.from_float(verdict['worst'])
                            - Fraction.from_float(verdict['L'])*Fraction.from_float(verdict['tau']))
        self.assertGreaterEqual(verdict['uniform_slack'], 0)
        self.assertLessEqual(Fraction.from_float(verdict['uniform_slack']), exact_expression)

    def test_boundary_with_negative_rounded_slack_is_inconclusive(self):
        smallest = np.nextafter(0., 1.)
        def boundary(*args, **kwargs):
            kwargs['values'][:] = -smallest
        with patch.object(bounds, 'lipschitz_certified', return_value=0.), \
                patch.object(cegis_c, 'lam_max_C_norm', return_value=np.full(9, -1.)), \
                patch.object(cegis_c, 'grid_enclosure', side_effect=boundary):
            verdict = cegis_c.verify(np.zeros((3, 3)), 3, normalized=True)
        self.assertFalse(verdict['certified'])
        self.assertTrue(verdict['grid_enclosed'])
        self.assertTrue(verdict['viol'].any())
        self.assertLess(verdict['uniform_slack'], 0.)
        self.assertGreater(verdict['worst']-verdict['margin'], 0.)

    def test_supplied_grid_cannot_silently_omit_domain_points(self):
        with self.assertRaisesRegex(ValueError, 'Cartesian grid'):
            cegis_c.grid_enclosure(np.zeros((3, 3)), 3, X=np.zeros((9, 2)))

    def test_residual_budget_rounds_in_the_sufficient_direction(self):
        # A synthetic certified raw grid upper bound isolates budget arithmetic.
        check = {'worst_upper': -1., 'tau': .01, 'grid_enclosed': True,
                 'outward_rounded': True}
        with patch.object(cegis_c, 'grid_enclosure', return_value=check):
            result = residual.residual_certified_constants(np.zeros((3, 3)), depth=1, n_grid_eta=9)
        self.assertTrue(result['budget_outward_rounded'])
        q = lambda x: Fraction.from_float(float(x))
        eta_expression = -q(result['worst_raw_grid']) - q(result['Lg'])*q(result['tau'])
        self.assertLessEqual(q(result['eta']), eta_expression)
        denominator = q(result['mu1'])*q(result['c_d0']) + 2*q(result['b_upper'])*q(result['c_d1'])
        self.assertLessEqual(q(result['delta_star'])*denominator, q(result['eta']))
        self.assertGreater(result['delta_star'], 0)


if __name__ == '__main__':
    unittest.main()
