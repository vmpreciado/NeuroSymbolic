"""Controls, practical bounded updates, and exact scale-chase evidence."""
import contextlib
import copy
from fractions import Fraction
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import bounds
import cegis_c
import experiments
import revision_experiments as revision


class ControlledProtocolTests(unittest.TestCase):
    def test_common_initialization_and_rng_are_seeded_and_local(self):
        protocol = revision.Protocol(initial_samples=30)
        theta, data, rng = revision.initial_state(8, protocol)
        theta2, data2, rng2 = revision.initial_state(8, protocol)
        np.testing.assert_array_equal(theta, experiments.identity_init())
        np.testing.assert_array_equal(theta, theta2)
        np.testing.assert_array_equal(data, data2)
        np.testing.assert_array_equal(rng.randn(12, 2), rng2.randn(12, 2))
        self.assertFalse(np.array_equal(data, revision.initial_state(9, protocol)[1]))

    def test_unbounded_learner_preserves_original_adam(self):
        config = revision.Protocol(initial_samples=40, training_steps=12)
        theta, data, _ = revision.initial_state(1, config)
        for normalized in (False, True):
            revised, metadata = revision.learn(theta, data, 0.05, normalized, config)
            original = cegis_c.adam(theta, data, 0.05, steps=12, lr=0.02,
                                    normalized=normalized)
            np.testing.assert_array_equal(revised, original)
            self.assertEqual(metadata['projection_steps'], 0)

    def test_projection_occurs_during_optimization_and_stays_in_ball(self):
        theta = experiments.identity_init()
        radius = float(np.linalg.norm(theta)) * 1.01
        config = revision.Protocol(initial_samples=100, training_steps=30,
                                   learning_rate=1.0, radius=radius)
        _, data, _ = revision.initial_state(0, config)
        learned, metadata = revision.learn(theta, data, 2.0, True, config)
        self.assertGreater(metadata['projection_steps'], 1)
        self.assertLessEqual(metadata['max_post_projection_norm'], radius + 2e-14)
        self.assertLessEqual(np.linalg.norm(learned), radius + 2e-14)
        self.assertLessEqual(Fraction(metadata['max_parameter_norm_squared_exact']),
                             Fraction.from_float(radius) ** 2)
        self.assertLessEqual(revision.exact_norm_squared(learned), Fraction.from_float(radius) ** 2)

    def test_exact_projection_detects_violation_hidden_by_rounded_norm(self):
        theta = np.zeros((3, 3))
        theta[0, 0], theta[0, 1] = 1., 2. ** -27
        self.assertEqual(np.linalg.norm(theta), 1.)
        self.assertGreater(revision.exact_norm_squared(theta), Fraction(1))
        projected = revision.project_ball(theta, 1.)
        self.assertLessEqual(revision.exact_norm_squared(projected), Fraction(1))
        self.assertFalse(np.array_equal(theta, projected))

    def test_all_means_all_margin_violators_not_entire_grid(self):
        verified = {'g': np.array([2., -3., 1., -4.]),
                    'viol': np.array([True, False, True, False])}
        np.testing.assert_array_equal(revision.select_margin_points(verified, 'all'), [0, 2])

    def test_rejected_metric_has_no_quantitative_certificate(self):
        self.assertEqual(revision.quantitative_outputs(experiments.identity_init(),
                                                       {'certified': False}, True, 3), {})
        with self.assertRaises(AssertionError):
            revision.quantitative_outputs(experiments.identity_init(),
                {'certified': True, 'grid_enclosed': False, 'uniform_slack': 1.}, True, 3)

    def test_zero_slack_has_zero_margin_and_base_rate(self):
        values = revision.quantitative_outputs(experiments.identity_init(),
            {'certified': True, 'grid_enclosed': True, 'uniform_slack': 0.}, True, 3)
        self.assertEqual(values['raw_margin_lower'], 0.)
        self.assertEqual(values['rate_lower'], cegis_c.RATE)

    def test_exact_polynomial_scaling_scales_floor_and_matrix(self):
        original = bounds.assemble_exact(experiments.identity_init())
        scaled = revision.scale_polynomials(original, 8.)
        for key in ('m11', 'm12', 'm22', 'c11', 'c12', 'c22', 'v'):
            self.assertEqual(scaled[key].coeffs,
                             {term: value * 8 for term, value in original[key].coeffs.items()})
        # Fixed-floor theta scaling would fail this exact identity.
        other = bounds.assemble_exact(np.sqrt(8.) * experiments.identity_init())
        self.assertNotEqual(scaled['m11'].coeffs, other['m11'].coeffs)


class CompletedPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with patch.object(revision, 'HERE', cls.root), contextlib.redirect_stdout(io.StringIO()):
            cls.result = revision.main(['--pilot'])

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_complete_pilot_passes_checks_without_overwriting_production(self):
        self.assertTrue(revision.validate_results(self.result))
        self.assertTrue((self.root/'results/revision_experiments_pilot.json').is_file())
        self.assertTrue((self.root/'results/timing_revision_pilot.json').is_file())
        self.assertFalse((self.root/'results/revision_experiments.json').exists())
        self.assertTrue((self.root/'paper/fig_ablation_pilot.pdf').is_file())
        self.assertTrue((self.root/'paper/fig_scale_chase_pilot.pdf').is_file())

    def test_missing_seed_is_rejected(self):
        bad = copy.deepcopy(self.result)
        del bad['regimes']['raw_fixed']['1']
        with self.assertRaisesRegex(ValueError, 'seed matrix'):
            revision.validate_results(bad)

    def test_changed_factorial_grid_is_rejected(self):
        bad = copy.deepcopy(self.result)
        bad['regimes']['normalized_coupled']['0']['protocol']['n_grid'] += 1
        with self.assertRaisesRegex(ValueError, 'grid difference'):
            revision.validate_results(bad)

    def test_changed_initial_dataset_is_rejected(self):
        bad = copy.deepcopy(self.result)
        bad['regimes']['raw_fixed']['0']['initial_dataset_sha256'] = 'altered'
        with self.assertRaisesRegex(ValueError, 'initialization'):
            revision.validate_results(bad)

    def test_norm_escape_cannot_be_hidden_by_final_projection(self):
        bad = copy.deepcopy(self.result)
        bad['regimes']['raw_coupled_bounded']['0']['trace'][0]['max_post_projection_norm'] = 1e4
        with self.assertRaisesRegex(ValueError, 'escaped'):
            revision.validate_results(bad)

    def test_constructive_chase_is_strict_valid_uncertified_and_scale_invariant(self):
        chase = self.result['scale_chase']
        self.assertGreater(chase['base_uniform_slack_lower'], 0.)
        self.assertGreater(chase['base_raw_gap_interval'][0], 0.)
        self.assertTrue(all(chase['checks'].values()))
        rows = chase['rows']
        self.assertEqual(len({row['normalized_gap_upper'] for row in rows}), 1)
        self.assertGreater(rows[-1]['raw_gap_lower'], rows[0]['raw_gap_lower'])
        self.assertIsNone(rows[0]['raw_training_margin'])
        for previous, current in zip(rows, rows[1:]):
            self.assertGreaterEqual(current['c'], 2 * previous['c'])
            self.assertEqual(current['raw_training_margin'], previous['next_training_margin_upper'])
            self.assertGreater(Fraction.from_float(current['c']) *
                               Fraction.from_float(chase['base_uniform_slack_lower']),
                               Fraction.from_float(current['raw_training_margin']))
            self.assertGreaterEqual(Fraction.from_float(previous['next_training_margin_upper']),
                Fraction.from_float(chase['gamma']) * Fraction.from_float(previous['raw_L_upper']) *
                Fraction.from_float(chase['tau_upper']))
        bad = copy.deepcopy(self.result)
        bad['scale_chase']['rows'][1]['normalized_gap_upper'] += 1
        with self.assertRaisesRegex(ValueError, 'normalized chase gap'):
            revision.validate_results(bad)


if __name__ == '__main__':
    unittest.main()
