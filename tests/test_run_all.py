"""Failure-path checks for the reproduction runner; no experiments are run."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_all
from rounding import mul_up, sub_down


class RunnerValidationTests(unittest.TestCase):
    def smoke(self):
        return {"candidate": {"certified": True, "grid_enclosed": True, "outward_rounded": True},
                "verification_backend": {"outward_rounded": True},
                "s": sub_down(0.3, mul_up(2.0, 0.1)),
                "L_cert": 2.0, "tau": 0.1, "worst_gt": -0.3}

    def benchmark(self):
        return {"n_seeds": 2, "certified": 1, "diverged": 1,
                "config": {"max_rounds": 2, "training_steps": 5, "initial_samples": 5,
                           "growth_factor": 100.0, "learning_rate": 0.01,
                           "base_margin": 0.05, "gamma": 1.3,
                           "optimizer": "Adam", "optimizer_state": "reset each retraining round",
                           "outcome_criterion": {"certified": "oracle", "diverged": "growth",
                                                 "stalled": "remaining uncertified"}},
                "verification_backend": {"outward_rounded": True},
                "validation": {"assembly": True, "enclosure": True},
                "example": {"certified": True, "L_cert": 2.0,
                            "slack": 0.3, "tau": 0.1, "grid_enclosed": True, "outward_rounded": True,
                            "uniform_slack": sub_down(0.3, mul_up(2.0, 0.1))},
                "runs": {"0": {"certified": True, "diverged": False, "trace": [{"gap": -0.1, "worst": -0.3, "L": 2.0, "tau": 0.1,
                                                "grid_enclosed": True, "outward_rounded": True,
                                                "uniform_slack": sub_down(0.3, mul_up(2.0, 0.1))}]},
                         "1": {"certified": False, "diverged": True,
                               "trace": [{"gap": 0.1, "L": 2.0}, {"gap": 0.2, "L": 201.0}]}}}

    def test_smoke_must_actually_certify_and_have_consistent_slack(self):
        run_all.validate_smoke(self.smoke())
        for change in ({"candidate": {"certified": False}}, {"s": -0.1},
                       {"s": 0.2}, {"s": float("nan")}):
            data = self.smoke()
            data.update(change)
            with self.subTest(change=change), self.assertRaises(run_all.ValidationError):
                run_all.validate_smoke(data)

    def test_benchmark_rejects_failed_checks_no_certificates_and_bad_margin(self):
        run_all.validate_benchmark(self.benchmark())
        mutations = [lambda d: d.update(certified=0),
                     lambda d: d.update(certified=2),
                     lambda d: d["validation"].update(enclosure=False),
                     lambda d: d.update(validation={}),
                     lambda d: d["example"].update(certified=False),
                     lambda d: d["example"].update(slack=0.01),
                     lambda d: d["runs"].pop("1"),
                     lambda d: d["runs"]["0"]["trace"][-1].update(gap=0.1),
                     lambda d: d["runs"]["1"]["trace"][-1].update(gap=float("nan"))]
        for mutate in mutations:
            data = self.benchmark()
            mutate(data)
            with self.subTest(data=data), self.assertRaises(run_all.ValidationError):
                run_all.validate_benchmark(data)

    def test_neural_coarse_summary_must_match_seed_outcomes(self):
        data = self.benchmark()
        data["certified_op"] = data.pop("certified")
        data["diverged_op"] = data.pop("diverged")
        data["certified_coarse"] = 1
        data["op_grid"] = 3
        data["coarse_grid"] = 2
        data["depth"] = 3
        data["config"].update(target_rate=0.4, metric_floor=0.1)
        outputs = {"a": 0.1, "b": 2.0, "metric_bounds_outward_rounded": True,
                   "metric_bounds_depth": 3, "raw_margin_lower": 0.01,
                   "rate_lower": 0.4, "overshoot_upper": 5.0}
        data["example"].update(outputs)
        data["runs"]["0"].update(outputs)
        data["example"]["gridpoints_covered"] = 9
        data["runs"]["0"]["trace"][-1]["gridpoints_covered"] = 9
        data["runs"]["0"].update(coarse_grid_enclosed=True, coarse_gridpoints_covered=4,
                                 coarse_uniform_slack=0.01)
        data["runs"]["0"]["certified_coarse"] = True
        data["runs"]["1"]["certified_coarse"] = False
        run_all.validate_benchmark(data, neural=True)
        data["certified_coarse"] = 0
        with self.assertRaisesRegex(run_all.ValidationError, "coarse-grid count"):
            run_all.validate_benchmark(data, neural=True)

    def test_protocol_rejects_missing_metadata_and_misclassified_outcomes(self):
        for mutate in (lambda d: d.pop("config"),
                       lambda d: d["config"].update(max_rounds=0),
                       lambda d: d["config"].update(growth_factor=1000.0),
                       lambda d: d["config"].update(outcome_criterion={})):
            data = self.benchmark()
            mutate(data)
            with self.subTest(data=data), self.assertRaises(run_all.ValidationError):
                run_all.validate_benchmark(data)

    def test_neural_quantitative_outputs_reject_unsafe_rounding_and_missing_bounds(self):
        config = {"target_rate": 0.4, "metric_floor": 0.1}
        outputs = {"a": 0.1, "b": 2.0, "metric_bounds_outward_rounded": True,
                   "metric_bounds_depth": 3, "raw_margin_lower": 0.01,
                   "rate_lower": 0.4, "overshoot_upper": 5.0}
        run_all.validate_neural_metric_outputs(outputs, 0.1, config, 3)
        for mutation in ({"b": None}, {"metric_bounds_outward_rounded": False},
                         {"metric_bounds_depth": 2}, {"raw_margin_lower": 0.03},
                         {"rate_lower": 0.41}, {"overshoot_upper": 4.0}):
            with self.subTest(mutation=mutation), self.assertRaises(run_all.ValidationError):
                run_all.validate_neural_metric_outputs(dict(outputs, **mutation), 0.1, config, 3)

    def test_residual_rejects_soundness_violation_and_nonfinite_output(self):
        data = {"eta": 0.1, "b_upper": 2.0, "delta_star": 0.01,
                "verification_backend": {"outward_rounded": True},
                "grid_enclosed": True, "budget_outward_rounded": True,
                "rows": [{"measured": -0.1, "bound": 0.0}]}
        run_all.validate_residual(data)
        for measured in (0.01, float("nan")):
            changed = copy.deepcopy(data)
            changed["rows"][0]["measured"] = measured
            with self.assertRaises(run_all.ValidationError):
                run_all.validate_residual(changed)

    def test_positive_results_require_outward_grid_and_downward_slack(self):
        for change in ({"grid_enclosed": False}, {"outward_rounded": False},
                       {"uniform_slack": 0.10000000000000001}):
            data = self.benchmark()
            data["example"].update(change)
            with self.subTest(change=change), self.assertRaises(run_all.ValidationError):
                run_all.validate_benchmark(data)
        data = self.benchmark()
        data.pop("verification_backend")
        with self.assertRaisesRegex(run_all.ValidationError, "outward-rounded"):
            run_all.validate_benchmark(data)

    def test_latex_rejects_undefined_citations_references_and_incomplete_log(self):
        completed = "Output written on paper.pdf (14 pages, 123 bytes).\n"
        pages, warnings = run_all.validate_latex_log(completed)
        self.assertEqual(pages, 14)
        self.assertEqual(warnings, [])
        for failure in ("LaTeX Warning: Citation `missing' on page 1 undefined",
                        "LaTeX Warning: Reference `missing' on page 2 undefined",
                        "LaTeX Warning: There were undefined references.",
                        "LaTeX Warning: Label `x' multiply defined.",
                        "Undefined control sequence.", "LaTeX Error: missing package"):
            with self.subTest(failure=failure), self.assertRaises(run_all.ValidationError):
                run_all.validate_latex_log(completed + failure)
        with self.assertRaises(run_all.ValidationError):
            run_all.validate_latex_log("This is an old or incomplete log")

    def test_layout_warnings_are_reported_without_becoming_false_errors(self):
        pages, warnings = run_all.validate_latex_log(
            "Overfull \\hbox (1.0pt too wide) in paragraph\n"
            "Output written on paper.pdf (14 pages, 123 bytes).\n")
        self.assertEqual(pages, 14)
        self.assertEqual(len(warnings), 1)

    def test_nonzero_process_exit_cannot_pass_by_printing_certified(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(run_all.ValidationError, "exited 3"):
                run_all.run_command([sys.executable, "-c",
                                     "print('certified 20/20'); raise SystemExit(3)"])

    def test_successful_process_cannot_reuse_stale_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stale.json"
            path.write_text("{}")
            os.utime(path, (1, 1))
            with patch.object(run_all, "HERE", Path(directory)), \
                    patch.object(run_all, "run_command", return_value="certified 20/20"):
                with self.assertRaisesRegex(run_all.ValidationError, "stale output"):
                    run_all.run_script("unused.py", outputs=("stale.json",))
            fresh = time.time_ns()
            path.write_text("{}")
            run_all.require_fresh([path], fresh)

    def test_latex_honors_return_code_even_when_old_good_log_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            paper = Path(directory)
            (paper / (run_all.TEX + ".log")).write_text(
                "Output written on paper.pdf (14 pages, 123 bytes).")
            with patch.object(run_all, "PAPER", paper), patch.object(
                    run_all, "run_command", side_effect=run_all.ValidationError("process exited 1")):
                with self.assertRaisesRegex(run_all.ValidationError, "exited 1"):
                    run_all.latex()

    def test_main_stops_after_failure_and_writes_manifest_and_full_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            later_called = []

            def fail():
                print("complete diagnostic output")
                raise run_all.ValidationError("certification failed")

            with patch.object(run_all, "HERE", root), patch.object(run_all, "RESULTS", results), \
                    patch.object(run_all, "environment", return_value={"test": True}), \
                    patch.object(run_all, "source_hashes", return_value={"source": "hash"}), \
                    patch.object(run_all, "artifact_hashes", return_value={}), \
                    patch.object(run_all, "stages", return_value=[
                        ("failing stage", fail), ("must not run", lambda: later_called.append(True))]), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_all.main(), 1)
            self.assertEqual(later_called, [])
            manifest = json.loads((results / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "fail")
            self.assertEqual(len(manifest["stages"]), 1)
            stage = manifest["stages"][0]
            self.assertEqual(stage["status"], "fail")
            stage_log = root / stage["log"]
            self.assertEqual(run_all.sha256(stage_log), stage["log_sha256"])
            self.assertIn("complete diagnostic output", stage_log.read_text())
            self.assertIn("complete diagnostic output", (results / "run_all.log").read_text())
            self.assertIn("Pipeline stopped", (results / "run_all.log").read_text())


if __name__ == "__main__":
    unittest.main()
