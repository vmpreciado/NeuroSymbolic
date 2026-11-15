"""Independent matrix/model checks and fail-closed constant-proof replay."""
import copy
from fractions import Fraction as F
import json
import unittest

import gmpy2 as g

import application_constant_baseline as baseline


class ConstantMetricBaselineTests(unittest.TestCase):
    def test_exact_matrix_identity_from_jacobian(self):
        P = ((F(1), F(1, 2)), (F(1, 2), F(2)))
        for k in (F(-1, 4), F(0), F(27, 100), F(1, 2), F(7, 10), F(1)):
            A = ((F(0), F(1)), (-k, F(-1, 2)))
            for rate in (F(1, 20), F.from_float(.05)):
                actual = tuple(tuple(sum(A[m][i]*P[m][j] + P[i][m]*A[m][j]
                                         for m in range(2)) + 2*rate*P[i][j]
                                     for j in range(2)) for i in range(2))
                self.assertEqual(actual, baseline.contraction_matrix(k, rate))

    def test_exact_small_box_minors_and_stronger_margin(self):
        row = baseline.certify_box(.25)
        self.assertEqual(row["cosine_bounds"]["coarse_interval_rational"], ["27/100", "7/10"])
        checks = row["decimal_target_check"]["endpoint_checks"]
        self.assertEqual([F(c["determinant_rational"]) for c in checks],
                         [F(6830301, 100000000), F(11986001, 100000000)])
        self.assertEqual([F(c["leading_principal_minor_rational"]) for c in checks],
                         [F(1699, 10000), F(5999, 10000)])
        self.assertTrue(row["shared_binary64_target_check"]["certified"])
        self.assertTrue(row["stronger_decimal_margin_check"]["certified"])
        self.assertGreater(baseline.SHARED_RATE, baseline.RATE)
        self.assertGreater(baseline.SHARED_MARGIN, baseline.MARGIN)

    def test_direct_high_precision_original_phase_and_matrix(self):
        # Separate 320-bit direct evaluation of the original shifted phase,
        # not the verifier's rational endpoint algebra or transformed sine.
        with g.context(precision=320):
            mp = g.mpfr
            phase = g.const_pi()/3
            P = ((mp(1), mp(.5)), (mp(.5), mp(2)))
            for B in baseline.BOXES:
                record = baseline.certify_box(B)
                lower, upper = map(F, record["cosine_bounds"]["mpfr_interval_rational"])
                lower = g.mpq(lower.numerator, lower.denominator)
                upper = g.mpq(upper.numerator, upper.denominator)
                for i in range(31):
                    delta = -mp(B) + 2*mp(B)*i/30
                    k = g.cos(delta+phase)
                    self.assertLessEqual(lower, k)
                    self.assertGreaterEqual(upper, k)
                    if B == .25:
                        A = ((mp(0), mp(1)), (-k, -mp(.5)))
                        C = tuple(tuple(sum(A[m][a]*P[m][b]+P[a][m]*A[m][b]
                                            for m in range(2))+2*mp(.05)*P[a][b]
                                        for b in range(2)) for a in range(2))
                        c11, c12, c22 = C[0][0], C[0][1], C[1][1]
                        eigmax = (c11+c22)/2+g.sqrt(((c11-c22)/2)**2+c12*c12)
                        self.assertLess(eigmax, -mp(.0001))

    def test_larger_boxes_have_actual_fixed_candidate_violations(self):
        with g.context(precision=320):
            for B in (.5, .75):
                row = baseline.certify_box(B)
                self.assertFalse(row["certified"])
                self.assertEqual(row["status"], "fixed_candidate_violates_requested_rate")
                bound = F(row["violation"]["rayleigh_quotient_lower_rational"])
                self.assertGreater(bound, 0)
                actual = 2*g.mpfr(.05) - g.cos(g.mpfr(B)+g.const_pi()/3)
                self.assertLessEqual(g.mpq(bound.numerator, bound.denominator), actual)
                self.assertIn("other constant", row["violation"]["interpretation"])

    def test_metric_bounds_and_prefactor(self):
        bounds = baseline.metric_bounds()
        self.assertEqual(bounds["lower_shift_principal_minors_rational"], ["1/4", "1/16"])
        self.assertEqual(bounds["upper_shift_principal_minors_rational"], ["5/4", "1/16"])
        self.assertGreaterEqual(F.from_float(bounds["overshoot_upper"])**2, 3)

    def test_full_json_roundtrip_and_every_domain(self):
        report = json.loads(json.dumps(baseline.build_report(), allow_nan=False))
        self.assertEqual(baseline.validate_report(report),
                         {"status":"pass", "domains":3, "certified_domains":1,
                          "fixed_candidate_violations":2})
        self.assertTrue(report["post_hoc"])
        self.assertFalse(report["training_performed"])
        self.assertIsNone(report["seeds"])

    def test_replay_rejects_tampered_targets_domain_proof_and_metadata(self):
        original = baseline.build_report()
        mutations = [
            lambda x: x.update(requested_rate=.04),
            lambda x: x.update(requested_raw_margin=.00001),
            lambda x: x["domains"][0].update(box_half_width=.5),
            lambda x: x["domains"][0]["decimal_target_check"]["endpoint_checks"][0].update(determinant_rational="1/1"),
            lambda x: x["domains"][1]["violation"].update(rayleigh_quotient_lower_rational="1/1"),
            lambda x: x["domains"].pop(),
            lambda x: x.update(post_hoc=False),
            lambda x: x.update(seeds=20),
            lambda x: x["verification_backend"].update(precision_bits=53),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(original)
            mutate(changed)
            with self.assertRaises(ValueError):
                baseline.validate_report(changed)

    def test_unprescribed_or_nonfinite_domain_is_rejected(self):
        for B in (0, .1, 1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                baseline.certify_box(B)


if __name__ == "__main__":
    unittest.main()
