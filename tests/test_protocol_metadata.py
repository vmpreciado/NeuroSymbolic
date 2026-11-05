"""Paper protocol tables reject missing budgets and ambiguous outcome settings."""
import unittest

import experiments
import fill_tex


class ProtocolPublicationTests(unittest.TestCase):
    def test_old_result_without_budget_metadata_cannot_fill_protocol_table(self):
        old_config = {"max_iter": 12, "gamma": 1.3, "eps0": .05}
        with self.assertRaisesRegex(ValueError, "max_rounds"):
            fill_tex.protocol_macros(old_config, "Poly")

    def test_disagreeing_round_cap_is_rejected(self):
        config = experiments.protocol_config()
        config["max_iter"] = config["max_rounds"] + 1
        with self.assertRaisesRegex(ValueError, "round caps disagree"):
            fill_tex.protocol_macros(config, "Poly")

    def test_incomplete_or_invalid_metric_bounds_cannot_enter_paper(self):
        for example in ({"a": .1}, {"a": .1, "b": float("inf")},
                        {"a": 0., "b": 2.}, {"a": 2., "b": 1.}):
            with self.subTest(example=example), self.assertRaisesRegex(ValueError, "metric bounds"):
                fill_tex.validate_metric_bounds(example, "neural")
        fill_tex.validate_metric_bounds({"a": .1, "b": 2.}, "neural")

    def test_growth_threshold_is_strict_and_certification_takes_precedence(self):
        factor = experiments.GROWTH_FACTOR
        for maximum, certified, expected in (
                (factor, False, "stalled"),
                (factor + 1, False, "diverged"),
                (factor + 1, True, "certified")):
            trace = [{"L": 1., "gap": 1.},
                     {"L": maximum, "gap": -1. if certified else 1.}]
            result = experiments.classify(trace, certified)
            self.assertTrue(result[expected])
            self.assertEqual(sum(result[key] for key in ("certified", "stalled", "diverged")), 1)


if __name__ == "__main__":
    unittest.main()
