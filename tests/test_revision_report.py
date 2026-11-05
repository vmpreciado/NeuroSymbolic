"""Certificate reporting must retain directed bounds and source provenance."""
from fractions import Fraction
import unittest
import numpy as np
import revision_report as report
import bounds
from rounding import mul_up, sub_down


class ReportingTests(unittest.TestCase):
    def test_trace_conversion_rate_and_overshoot_are_safe(self):
        theta = np.zeros((3, 3))
        theta[0, 0] = theta[2, 0] = 1.0
        run = dict(certified=True, theta=theta.tolist(), trace=[dict(
            grid_enclosed=True, outward_rounded=True, L=2.0, tau=.001, worst=-.04,
            uniform_slack=sub_down(.04, mul_up(2., .001)))])
        row=report.certificate_row("Polynomial", "0", run, {}, 2001, .4, 2.)
        F=Fraction.from_float
        self.assertLessEqual(F(row["eta"]), F(row["trace_lower"])*F(row["uniform_slack"]))
        self.assertLessEqual(F(row["rate_lower"]), F(.4)+F(row["eta"])/(2*F(row["b"])))
        self.assertGreaterEqual(F(row["overshoot_upper"])**2,F(row["b"])/F(row["a"]))
        self.assertGreater(row["trace_lower"],2*row["a"])

    def test_initial_verification_is_excluded_from_round_timing(self):
        data=dict(runs={"0":[dict(iter=0,train_s=0.,verify_including_bounds_s=3.),
                              dict(iter=1,train_s=.1,verify_including_bounds_s=.5)]})
        self.assertEqual(len(report.timing_rows(data)),1)
        self.assertEqual(report.timing_rows(data)[0]["iter"],1)

    def test_inconclusive_candidate_cannot_enter_certificate_table(self):
        with self.assertRaisesRegex(ValueError,"positive outward"):
            report.certificate_row("Polynomial",0,dict(certified=False,trace=[dict(grid_enclosed=False)]),
                                   {},2001,.4,2.)

    def test_nominal_or_inflated_slack_cannot_be_relabelled_as_certified(self):
        theta=np.zeros((3,3)); theta[0,0]=theta[2,0]=1.
        for change in ({"outward_rounded":False},{"uniform_slack":.04}):
            row=dict(grid_enclosed=True,outward_rounded=True,L=2.,tau=.001,worst=-.04,
                     uniform_slack=sub_down(.04,mul_up(2.,.001)))
            row.update(change)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,"positive outward"):
                report.certificate_row("Polynomial",0,dict(certified=True,theta=theta.tolist(),trace=[row]),
                                       {},2001,.4,2.)


if __name__ == "__main__": unittest.main()
