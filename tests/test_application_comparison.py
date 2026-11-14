"""Matched-design and evidence-integrity checks without production reruns.

Synthetic two-seed records use a genuinely replayable small-domain certificate;
their optimizer histories are fixtures, not scientific experimental results.
"""
import copy
import hashlib
import unittest
from unittest.mock import patch

import numpy as np

import application_comparison as study
import application_verifier as verifier
import bench_neural
from rounding import backend_info
from tests.test_application_verifier import constant_factor


def fixture():
    cfg = study.config()
    cfg.update(seeds=[0,1],domain_halfwidths=[.125],initial_samples=8,
               batch_size=8,max_rounds=1,training_steps_per_round=2,
               screen_axis_points=3,verification_max_boxes=127,
               verification_max_depth=8)
    cfg['secondary_refinement']['max_boxes']=255
    B=.125
    theta=constant_factor()
    cert=verifier.adaptive_oracle(theta,[-B]*2,[B]*2,rate=cfg['target_rate'],
        alpha=cfg['metric_floor'],target_margin=cfg['verification_raw_margin'],
        max_boxes=cfg['verification_max_boxes'],max_depth=cfg['verification_max_depth'],
        check_points=study.screening_grid(B,cfg['screen_axis_points']))
    assert cert['certified']
    runs=[]
    for seed in cfg['seeds']:
        initial=bench_neural.init_params(seed,cfg['initialization_scale'])
        data=np.random.default_rng(seed+100_000).uniform(-B,B,(cfg['initial_samples'],2))
        for method in cfg['methods']:
            zero={k:0 for k in cert['counters']}
            zero['nominal_point_evaluations']=cfg['screen_axis_points']**2
            trace=[{'round':0,'steps_completed':0,'dataset_size':8,'loss':1.,
                    'certified':False,'reason':'nominal_screen_inconclusive',
                    'counters':zero,'oracle_s':.001,'parameter_l2':float(np.linalg.norm(initial))},
                   {'round':1,'steps_completed':2,'dataset_size':8,'loss':0.,
                    'certified':True,'reason':'all_leaves_certified',
                    'counters':copy.deepcopy(cert['counters']),'oracle_s':.002,
                    'parameter_l2':float(np.linalg.norm(theta))}]
            runs.append({'method':method,'seed':seed,'halfwidth':B,
                         'initial_theta':initial.tolist(),
                         'initial_data_sha256':hashlib.sha256(data.tobytes()).hexdigest(),
                         'theta':theta.tolist(),'certified':True,'failure_reason':None,
                         'steps_completed':2,'learner_s':.01,'oracle_s':.003,'total_s':.02,
                         'trace':trace,'certificate':copy.deepcopy(cert),
                         'guarantees':verifier.metric_guarantees(theta,[-B]*2,[B]*2,
                             cfg['target_rate'],cfg['verification_raw_margin'],cfg['metric_floor'])})
    for run in runs:
        run['refinement']=study.refine_frozen_candidate(run,cfg)
    report={'schema_version':1,'config':cfg,'protocol_sha256':study.digest(study.PROTOCOL),
            'verification_backend':backend_info(),'baseline':study.baseline.protocol_description(),
            'runs':runs,'summary':study.summarize(runs,cfg),
            'energy_subsets':{str(B):verifier.certified_energy_level([-B]*2,[B]*2)},
            'production_run':True}
    return cfg,report


class ApplicationComparisonTests(unittest.TestCase):
    def setUp(self):
        self.cfg,self.report=fixture()
        self.config_patch=patch.object(study,'config',return_value=self.cfg)
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)

    def reject(self,mutate,replay=True):
        report=copy.deepcopy(self.report)
        mutate(report)
        with self.assertRaises((ValueError,KeyError)):
            study.validate_results(report,replay=replay)

    def test_real_certificate_fixture_replays_and_starts_are_matched(self):
        self.assertTrue(study.validate_results(self.report))
        for seed in self.cfg['seeds']:
            same=[r for r in self.report['runs'] if r['seed']==seed]
            self.assertEqual(len({r['initial_data_sha256'] for r in same}),1)
            for r in same[1:]:
                np.testing.assert_array_equal(r['initial_theta'],same[0]['initial_theta'])

    def test_driver_uses_identical_initial_candidates_and_samples(self):
        cfg=copy.deepcopy(self.cfg);cfg['max_rounds']=0
        def no_certificate(theta,lower,upper,**kwargs):
            return {'certified':False,'reason':'nominal_screen_inconclusive',
                    'training_points':[],
                    'counters':{'boxes_evaluated':0,'accepted_leaves':0,'excluded_leaves':0,
                    'splits':0,'maximum_depth_reached':0,
                    'nominal_point_evaluations':len(kwargs['check_points'])}}
        with patch.object(verifier,'adaptive_oracle',side_effect=no_certificate):
            runs=[study.run_one(m,0,.125,cfg) for m in cfg['methods']]
        self.assertEqual(len({r['initial_data_sha256'] for r in runs}),1)
        for r in runs:
            np.testing.assert_array_equal(r['initial_theta'],runs[0]['initial_theta'])
            self.assertEqual(r['steps_completed'],0)
            self.assertFalse(r['certified'])

    def test_missing_duplicate_unplanned_runs_are_rejected(self):
        self.reject(lambda d:d['runs'].pop())
        self.reject(lambda d:d['runs'].append(copy.deepcopy(d['runs'][0])))
        self.reject(lambda d:d['runs'][0].update(seed=1000))
        self.reject(lambda d:d['runs'][0].update(halfwidth=.2))
        self.reject(lambda d:d['runs'][0].update(method='undeclared_method'))

    def test_pilot_changed_protocol_or_baseline_cannot_enter_paper(self):
        self.reject(lambda d:d.update(production_run=False))
        self.reject(lambda d:d['config'].update(training_raw_margin=.001))
        self.reject(lambda d:d.update(protocol_sha256='0'*64))
        self.reject(lambda d:d['baseline'].update(loss='different method'))

    def test_unmatched_initialization_or_collocations_are_rejected(self):
        self.reject(lambda d:d['runs'][0]['initial_theta'].__setitem__(0,123.))
        self.reject(lambda d:d['runs'][0].update(initial_data_sha256='0'*64))
        # PDI and fixed-data controls must never receive verifier diagnostics.
        index=next(i for i,r in enumerate(self.report['runs']) if r['method']=='fitzsimmons_liu_pdi')
        self.reject(lambda d:d['runs'][index]['trace'][1].update(dataset_size=9))

    def test_unequal_or_exceeded_budgets_are_rejected(self):
        self.reject(lambda d:d['runs'][0].update(steps_completed=3))
        self.reject(lambda d:d['runs'][0]['trace'].append(copy.deepcopy(d['runs'][0]['trace'][-1])))
        self.reject(lambda d:d['runs'][0]['trace'][0]['counters'].update(boxes_evaluated=128))
        self.reject(lambda d:d['runs'][0]['trace'][0]['counters'].update(nominal_point_evaluations=25))
        self.reject(lambda d:d['runs'][0]['certificate']['budget'].update(max_boxes=1000))

    def test_wrong_candidate_floor_region_or_margin_is_rejected(self):
        self.reject(lambda d:d['runs'][0]['theta'].__setitem__(-1,2.))
        self.reject(lambda d:d['runs'][0]['certificate'].update(alpha=.2))
        self.reject(lambda d:d['runs'][0]['certificate'].update(quadratic_region={'matrix':[[1,0],[0,1]],'level':.01}))
        self.reject(lambda d:d['runs'][0]['certificate'].update(target_margin=0.))
        self.reject(lambda d:d['runs'][0]['certificate'].update(domain_upper=[.1,.1]))

    def test_failed_replay_or_false_guarantee_is_rejected(self):
        self.reject(lambda d:d['runs'][0]['certificate']['proof']['leaf_paths'].pop())
        self.reject(lambda d:d['runs'][0]['guarantees'].update(rate_lower=10.))
        self.reject(lambda d:d['energy_subsets']['0.125'].update(rho_lower=10.))
        self.reject(lambda d:d['summary']['rows'][0].update(certified=200))

    def test_timing_accounting_and_early_failure_are_rejected(self):
        self.reject(lambda d:d['runs'][0].update(oracle_s=.1))
        self.reject(lambda d:d['runs'][0].update(total_s=.001))
        def early_failure(d):
            r=d['runs'][0]
            r.update(certified=False,certificate=None,failure_reason='nominal_screen_inconclusive',steps_completed=0,oracle_s=.001)
            r['trace']=r['trace'][:1]
        self.reject(early_failure)

    def test_refinement_freezes_every_candidate_and_changes_only_box_cap(self):
        run=copy.deepcopy(self.report['runs'][0])
        before=copy.deepcopy(run)
        original=verifier.adaptive_oracle
        with patch.object(verifier,'adaptive_oracle',wraps=original) as oracle:
            result=study.refine_frozen_candidate(run,self.cfg)
        self.assertEqual(run,before)
        self.assertEqual(oracle.call_count,1)
        args,kwargs=oracle.call_args
        np.testing.assert_array_equal(args[0],run['theta'])
        self.assertEqual(args[1],[-.125]*2)
        self.assertEqual(args[2],[.125]*2)
        self.assertEqual(kwargs['rate'],self.cfg['target_rate'])
        self.assertEqual(kwargs['alpha'],self.cfg['metric_floor'])
        self.assertEqual(kwargs['target_margin'],self.cfg['verification_raw_margin'])
        self.assertEqual(kwargs['max_depth'],self.cfg['verification_max_depth'])
        self.assertEqual(kwargs['max_boxes'],self.cfg['secondary_refinement']['max_boxes'])
        np.testing.assert_array_equal(kwargs['check_points'],study.screening_grid(.125,3))
        self.assertTrue(result['selected_after_preliminary_outcomes'])
        self.assertTrue(result['exploratory'])
        self.assertEqual(result['total_primary_plus_refinement_s'],run['total_s']+result['oracle_s'])

    def test_refinement_missing_changed_or_relabelled_is_rejected(self):
        self.reject(lambda d:d['runs'][0].pop('refinement'))
        self.reject(lambda d:d['runs'][0]['refinement'].update(exploratory=False))
        self.reject(lambda d:d['runs'][0]['refinement'].update(selected_after_preliminary_outcomes=False))
        self.reject(lambda d:d['runs'][0]['refinement'].update(candidate_sha256='0'*64))
        self.reject(lambda d:d['runs'][0]['refinement']['certificate'].update(domain_upper=[.1,.1]))
        self.reject(lambda d:d['runs'][0]['refinement']['certificate'].update(target_margin=0.))
        self.reject(lambda d:d['runs'][0]['refinement']['certificate']['budget'].update(max_boxes=511))
        self.reject(lambda d:d['runs'][0]['refinement'].update(total_primary_plus_refinement_s=0.))

    def test_every_primary_and_refined_positive_proof_is_replayed(self):
        original=verifier.replay_oracle_certificate
        with patch.object(verifier,'replay_oracle_certificate',wraps=original) as replay:
            self.assertTrue(study.validate_results(self.report))
        self.assertEqual(replay.call_count,2*len(self.report['runs']))
        self.reject(lambda d:d['runs'][0]['refinement']['certificate']['proof']['leaf_paths'].pop())
        self.reject(lambda d:d['runs'][0]['refinement']['guarantees'].update(rate_lower=10.))
        self.reject(lambda d:d['summary']['refinement_rows'][0].update(certified=999))

    def test_primary_success_cannot_be_lost_in_refinement(self):
        def lose_success(d):
            refinement=d['runs'][0]['refinement']
            refinement['certified']=False
            refinement['certificate'].update(certified=False,status='inconclusive')
        self.reject(lose_success)


if __name__=='__main__':unittest.main()
