"""Independent matrix, protocol, endpoint, and fail-closed replay checks."""
import copy
from fractions import Fraction as F
import json
import unittest

import numpy as np

import cegis_c
import polynomial_constant_baseline as baseline


class PolynomialConstantMetricTests(unittest.TestCase):
    def test_matches_existing_benchmark_domain_rate_and_jacobian(self):
        self.assertEqual(F.from_float(cegis_c.RATE), baseline.SHARED_RATE)
        self.assertEqual(F.from_float(cegis_c.BOX), baseline.BOX)
        X = np.array([[-2., -2.], [-1., 2.], [0., 0.], [1., -2.], [2., 2.]])
        A = cegis_c.jac(X)
        f = cegis_c.f(X)
        for i, (x1, x2) in enumerate(X):
            np.testing.assert_array_equal(A[i], [[-1., 0.], [2*x1, -1.]])
            np.testing.assert_array_equal(f[i], [-x1, -x2+x1*x1])

    def test_exact_matrix_identity_from_independent_products(self):
        P = ((F(12), F(0)), (F(0), F(1)))
        for x1 in (F(-2), F(-11, 8), F(0), F(3, 7), F(2)):
            A = ((F(-1), F(0)), (2*x1, F(-1)))
            for rate in (F(2, 5), F.from_float(.4)):
                actual = tuple(tuple(sum(A[m][i]*P[m][j] + P[i][m]*A[m][j]
                                         for m in range(2)) + 2*rate*P[i][j]
                                     for j in range(2)) for i in range(2))
                self.assertEqual(actual, baseline.contraction_matrix(x1, rate))

    def test_exact_decimal_endpoint_minors_and_independent_margin(self):
        report = baseline.build_report()
        check = report["decimal_target_check"]
        self.assertEqual(check["raw_margin_rational"], "1/20")
        self.assertEqual(check["normalized_margin_rational"], "1/260")
        for row in check["endpoint_checks"]:
            self.assertEqual(F(row["leading_principal_minor_rational"]), F(72, 5))
            self.assertEqual(F(row["determinant_rational"]), F(32, 25))
            self.assertEqual(F(row["shifted_leading_principal_minor_rational"]), F(287, 20))
            self.assertEqual(F(row["shifted_determinant_rational"]), F(201, 400))
            self.assertTrue(row["shifted_positive_definite"])

    def test_diagonal_choice_threshold_from_independent_matrix_products(self):
        choice = baseline.build_report()["diagonal_choice"]
        threshold = F(choice["threshold_rational"])
        selected = F(choice["selected_p_rational"])
        self.assertEqual(threshold, F(100, 9))
        self.assertEqual(selected - threshold, F(choice["selection_gap_rational"]))
        # At the derived boundary the unshifted matrix is singular, so the
        # strict inequality matters. The selected integer has positive minors.
        # Reconstruct A'P+PA+2*rate*P without using the production formula.
        for p, expected_det in ((threshold - 1, F(-36, 25)),
                                (threshold, F(0)), (selected, F(32, 25))):
            metric = ((p, F(0)), (F(0), F(1)))
            for x1 in (F(-2), F(2)):
                A = ((F(-1), F(0)), (2*x1, F(-1)))
                negative = tuple(tuple(-sum(A[m][i]*metric[m][j] + metric[i][m]*A[m][j]
                                             for m in range(2)) - F(4, 5)*metric[i][j]
                                         for j in range(2)) for i in range(2))
                determinant = negative[0][0]*negative[1][1] - negative[0][1]*negative[1][0]
                self.assertGreater(negative[0][0], 0)
                self.assertEqual(determinant, expected_det)
        self.assertEqual(F(choice["endpoint_determinant_at_threshold_rational"]), 0)
        self.assertEqual(F(choice["endpoint_determinant_at_selected_p_rational"]), F(32, 25))
        self.assertIn("unshifted strict contraction", choice["margin_scope"])

    def test_shared_binary64_rate_is_distinct_and_also_certified(self):
        report = baseline.build_report()
        check = report["shared_binary64_target_check"]
        self.assertGreater(F(check["rate_rational"]), F(report["decimal_rate_rational"]))
        self.assertEqual(F(check["rate_rational"]), F(3602879701896397, 9007199254740992))
        self.assertTrue(check["certified"])
        for row in check["endpoint_checks"]:
            self.assertEqual(F(row["shifted_determinant_rational"]),
                             F(63699442661468368954797447970819,
                               126765060022822940149670320537600))
            self.assertGreater(F(row["shifted_leading_principal_minor_rational"]), 0)
            self.assertGreater(F(row["shifted_determinant_rational"]), 0)

    def test_affine_extension_identity(self):
        # The certificate uses this affine identity plus convexity of the
        # positive-definite cone. These exact evaluations independently catch
        # a missing/non-affine matrix term; they are not a sampled proof.
        for rate in (F(2, 5), F.from_float(.4)):
            left = baseline.contraction_matrix(-2, rate)
            right = baseline.contraction_matrix(2, rate)
            for t in (F(0), F(1, 17), F(1, 2), F(19, 20), F(1)):
                matrix = baseline.contraction_matrix(-2 + 4*t, rate)
                combination = tuple(tuple((1-t)*left[i][j]+t*right[i][j]
                                          for j in range(2)) for i in range(2))
                self.assertEqual(matrix, combination)

    def test_exact_metric_bounds_and_outward_prefactor(self):
        report = baseline.build_report()
        self.assertEqual(report["metric_positive_principal_minors_rational"], ["12/1", "12/1"])
        bounds = report["metric_bounds"]
        self.assertEqual(bounds["lower_shift_diagonal_rational"], ["11/1", "0/1"])
        self.assertEqual(bounds["upper_shift_diagonal_rational"], ["0/1", "11/1"])
        self.assertGreaterEqual(F.from_float(bounds["overshoot_upper"])**2, 12)
        self.assertEqual(F.from_float(bounds["overshoot_upper"]), F(bounds["overshoot_upper_rational"]))

    def test_json_roundtrip_and_scope(self):
        report = json.loads(json.dumps(baseline.build_report(), allow_nan=False))
        self.assertEqual(baseline.validate_report(report),
                         {"status": "pass", "certified": True, "target_conventions": 2,
                          "endpoint_checks_per_convention": 2})
        self.assertTrue(report["post_hoc"])
        self.assertFalse(report["training_performed"])
        self.assertIsNone(report["seeds"])
        self.assertIn("not forward invariant", report["trajectory_scope"])
        self.assertIn("not the learned candidates' training margins", report["margin_scope"])

    def test_replay_rejects_tampered_proof_protocol_and_scope(self):
        original = baseline.build_report()
        mutations = [
            lambda x: x.update(requested_rate=.3),
            lambda x: x["domain"].update(box_half_width=1.),
            lambda x: x["metric_rational"][0].__setitem__(0, "11/1"),
            lambda x: x["diagonal_choice"].update(threshold_rational="11/1"),
            lambda x: x["diagonal_choice"].update(strict_contraction_condition="p >= B^2/(1-lambda)^2"),
            lambda x: x["diagonal_choice"].update(margin_scope="criterion alone proves the requested raw margin"),
            lambda x: x["decimal_target_check"]["endpoint_checks"][0].update(determinant_rational="1/1"),
            lambda x: x["shared_binary64_target_check"]["endpoint_checks"][1].update(shifted_determinant_rational="1/1"),
            lambda x: x["decimal_target_check"]["endpoint_checks"].pop(),
            lambda x: x.update(raw_margin_lower_rational="1/10"),
            lambda x: x.update(post_hoc=False),
            lambda x: x.update(certified=1),
            lambda x: x.update(seeds=20),
            lambda x: x.update(trajectory_scope="entire box is invariant"),
            lambda x: x["metric_bounds"].update(overshoot_upper=float("nan")),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(original)
                mutate(changed)
                with self.assertRaises(ValueError):
                    baseline.validate_report(changed)


if __name__ == "__main__":
    unittest.main()
