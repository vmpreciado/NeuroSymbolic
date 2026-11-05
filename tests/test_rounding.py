"""Adversarial exact-rational and high-precision checks of outward arithmetic."""
from fractions import Fraction
import math
import unittest
from unittest.mock import patch

import gmpy2
import numpy as np

import rounding as r


def frac(x):
    return Fraction.from_float(float(x))


class ElementaryRoundingTests(unittest.TestCase):
    def assertContains(self, interval, exact):
        self.assertLessEqual(frac(interval.lo), exact)
        self.assertGreaterEqual(frac(interval.hi), exact)

    def test_elementary_point_operations_enclose_exact_rationals(self):
        tiny = np.nextafter(0.0, 1.0)
        cases = [(1.0, 2.0**-53), (1.0, -1.0), (tiny, 0.5),
                 (np.finfo(float).tiny, 0.5), (1e200, 1e-200),
                 (-1e200, 1e-200), (2.0**53, 1.0), (tiny, -tiny),
                 (1.0000000000000002, -1.0)]
        rng = np.random.default_rng(721)
        cases += [(float(rng.uniform(-1, 1)*10.0**rng.integers(-120, 121)),
                   float(rng.uniform(-1, 1)*10.0**rng.integers(-120, 121)))
                  for _ in range(250)]
        for a, b in cases:
            with self.subTest(a=a, b=b):
                x, y = r.Interval(a), r.Interval(b)
                self.assertContains(x+y, frac(a)+frac(b))
                self.assertContains(x-y, frac(a)-frac(b))
                self.assertContains(x*y, frac(a)*frac(b))
                if b and abs(a/b) < np.finfo(float).max:
                    self.assertContains(x/y, frac(a)/frac(b))

    def test_signed_interval_products_and_divisions(self):
        for lo, hi in [(-2., -1.), (-2., 3.), (0., 4.), (1., 3.)]:
            a = r.Interval(lo, hi)
            for blo, bhi in [(-7., -2.), (0.5, 5.)]:
                b = r.Interval(blo, bhi)
                prod, quotient = a*b, a/b
                for x in np.linspace(lo, hi, 9):
                    for y in np.linspace(blo, bhi, 9):
                        self.assertContains(prod, frac(x)*frac(y))
                        self.assertContains(quotient, frac(x)/frac(y))

    def test_squares_powers_and_subnormals(self):
        self.assertEqual(float(r.Interval(-2., 3.).square().lo), 0.0)
        for x in [-2., -np.nextafter(0., 1.), 0., np.nextafter(0., 1.), 3.]:
            for power in range(7):
                self.assertContains(r.Interval(x)**power, frac(x)**power)
        self.assertGreater(r.mul_up(np.nextafter(0., 1.), 0.5), 0.0)
        self.assertLess(r.mul_down(-np.nextafter(0., 1.), 0.5), 0.0)

    def test_outward_conversions_do_not_round_coefficients_inward(self):
        for q in [Fraction(1, 10), Fraction(-1, 10), Fraction(1, 2**2000),
                  Fraction(-1, 2**2000), Fraction(1)+Fraction(1, 2**100),
                  Fraction(-1)-Fraction(1, 2**100)]:
            self.assertLessEqual(frac(r.float_down(q)), q)
            self.assertGreaterEqual(frac(r.float_up(q)), q)

    def test_nonfinite_overflow_and_invalid_domains_fail_closed(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ArithmeticError): r.Interval(bad)
        with self.assertRaises(ArithmeticError): r.Interval(np.finfo(float).max)*2
        with self.assertRaises(ZeroDivisionError): r.Interval(1.)/r.Interval(-1., 1.)
        with self.assertRaises(ValueError): r.Interval(-2., -1.).sqrt()
        with self.assertRaises(ValueError): r.Interval(2., 1.)

    def test_sqrt_contains_directed_high_precision_reference(self):
        samples = [np.nextafter(0., 1.), 0., 2.0**-1000, 0.1, 2., 1e200]
        for x in samples:
            interval = r.Interval(x).sqrt()
            with gmpy2.context(precision=320, round=gmpy2.RoundDown):
                lower = gmpy2.sqrt(gmpy2.mpfr(float(x)))
            with gmpy2.context(precision=320, round=gmpy2.RoundUp):
                upper = gmpy2.sqrt(gmpy2.mpfr(float(x)))
            self.assertLessEqual(gmpy2.mpq(float(interval.lo)), lower)
            self.assertGreaterEqual(gmpy2.mpq(float(interval.hi)), upper)


class TranscendentalRoundingTests(unittest.TestCase):
    def assertReferenceEnclosed(self, interval, x, name):
        with gmpy2.context(precision=320, round=gmpy2.RoundDown):
            lower = getattr(gmpy2, name)(gmpy2.mpfr(float(x)))
        with gmpy2.context(precision=320, round=gmpy2.RoundUp):
            upper = getattr(gmpy2, name)(gmpy2.mpfr(float(x)))
        self.assertLessEqual(gmpy2.mpq(float(interval.lo)), lower)
        self.assertGreaterEqual(gmpy2.mpq(float(interval.hi)), upper)

    def test_mpfr_endpoint_paths_enclose_critical_points_and_tails(self):
        points = [-1000., -16., -math.pi, -math.pi/2, -1., -0.,
                  np.nextafter(0., 1.), 1., math.pi/2, math.pi, 16., 1000.]
        for name in ("tanh", "sin", "cos"):
            for x in points:
                with self.subTest(name=name, x=x):
                    self.assertReferenceEnclosed(getattr(r.Interval(x), name)(), x, name)

    def test_rigorous_interpolation_encloses_nodes_interiors_and_tails(self):
        rng = np.random.default_rng(993)
        h = r._TABLE_STEP
        nodes = np.array([-16., -h, 0., h, 16.])
        points = np.concatenate([nodes, np.nextafter(nodes, -np.inf),
                                 np.nextafter(nodes, np.inf),
                                 rng.uniform(-16., 16., 170), [-1000., 1000.]])
        # Force the same table path used for million-point grid blocks.
        with patch.object(r, "_SMALL_BATCH", 0):
            for name in ("tanh", "sin", "cos"):
                out = getattr(r.Interval(points), name)()
                for j, x in enumerate(points):
                    self.assertReferenceEnclosed(r.Interval(out.lo[j], out.hi[j]), x, name)

    def test_general_trig_intervals_contain_interior_extrema(self):
        for name in ("sin", "cos"):
            for lo, hi in [(-2., 2.), (0., math.pi), (-10., -7.), (1., 1.+2.0**-50)]:
                out = getattr(r.Interval(lo, hi), name)()
                for x in np.linspace(lo, hi, 31):
                    self.assertReferenceEnclosed(out, x, name)

    def test_large_batch_tanh_center_radius_contains_full_intervals(self):
        lower = np.array([-1000., -20., -16., -1., -1e-10, 0., 1., 16., 20., -1000.])
        upper = np.array([-999., -19., -15.99, 1., 1e-10, np.nextafter(0., 1.),
                          np.nextafter(1., 2.), 17., 21., 1000.])
        with patch.object(r, "_SMALL_BATCH", 0):
            actual = r.Interval(lower, upper).tanh()
        for j, (lo, hi) in enumerate(zip(lower, upper)):
            interval = r.Interval(actual.lo[j], actual.hi[j])
            for x in np.linspace(lo, hi, 17):
                self.assertReferenceEnclosed(interval, x, "tanh")


class CertificateHelpersTests(unittest.TestCase):
    def test_cover_radius_uses_largest_actual_gap(self):
        nodes = np.array([-2., -1.99, -0.7, 2.])
        tau = r.grid_radius(nodes)
        exact_gap = max(frac(b)-frac(a) for a, b in zip(nodes[:-1], nodes[1:]))
        with gmpy2.context(precision=320, round=gmpy2.RoundUp):
            reference = (gmpy2.mpfr(exact_gap.numerator)/exact_gap.denominator
                         *gmpy2.sqrt(gmpy2.mpfr(2))/2)
        self.assertGreaterEqual(gmpy2.mpq(tau), reference)
        self.assertGreater(tau, 1.)
        for invalid in [[0.], [0., 0., 1.], [0., 2., 1.]]:
            with self.assertRaises(ValueError): r.grid_radius(invalid)

    def test_eigenvalue_encloses_high_precision_symmetric_result(self):
        for a, b, c in [(1., 0., 1.), (-1., 1e-15, -1.), (1e100, -2e99, 3e100)]:
            out = r.lambda_max_2x2(a, b, c)
            with gmpy2.context(precision=320):
                aa, bb, cc = map(gmpy2.mpfr, (a, b, c))
                reference = (aa+cc)/2+gmpy2.sqrt(((aa-cc)/2)**2+bb**2)
            self.assertLessEqual(gmpy2.mpq(float(out.lo)), reference)
            self.assertGreaterEqual(gmpy2.mpq(float(out.hi)), reference)

    def test_backend_metadata_and_ieee_checks(self):
        self.assertTrue(r.self_test())
        self.assertIs(r.backend_info()["outward_rounded"], True)
        self.assertIn("MPFR", r.backend_info()["mpfr_version"])


if __name__ == "__main__":
    unittest.main()
