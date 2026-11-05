"""Independent high-precision checks of the rounded benchmark certificate paths."""
from fractions import Fraction
import json
import contextlib
import io
import tempfile
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import gmpy2 as gp
import numpy as np

import bench_neural as neural
import bench_pendulum as pendulum
from rounding import Interval


def _mp(value):
    if isinstance(value, Fraction):
        return gp.mpfr(value.numerator)/value.denominator
    return gp.mpfr(float(value))


def _metric_ref(ls, dl, d2l=None):
    a,b,c=ls
    M=(_mp(neural.ALPHA)+a*a+b*b,b*c,_mp(neural.ALPHA)+c*c)
    dm=[]
    for ak,bk,ck in dl:
        dm.append((2*a*ak+2*b*bk,bk*c+b*ck,2*c*ck))
    d2m={}
    if d2l is not None:
        for k in range(2):
            for l in range(2):
                ak,bk,ck=dl[k];al,bl,cl=dl[l];akl,bkl,ckl=d2l[k,l]
                d2m[k,l]=(2*ak*al+2*a*akl+2*bk*bl+2*b*bkl,
                           bkl*c+bk*cl+bl*ck+b*ckl,2*ck*cl+2*c*ckl)
    return M,dm,d2m


def _neural_ref(parameters,state):
    W1,b1,W2,b2=neural._unpack(parameters)
    x,y=map(_mp,state)
    h=[gp.tanh(_mp(W1[j,0])*x+_mp(W1[j,1])*y+_mp(b1[j])) for j in range(neural.H_UNITS)]
    hp=[1-t*t for t in h];hpp=[-2*t*u for t,u in zip(h,hp)]
    ls=[sum((_mp(W2[r,j])*h[j] for j in range(neural.H_UNITS)),_mp(b2[r])) for r in range(3)]
    dl=[[sum((_mp(W2[r,j])*_mp(W1[j,k])*hp[j] for j in range(neural.H_UNITS)),gp.mpfr(0)) for r in range(3)] for k in range(2)]
    d2l={(k,l):[sum((_mp(W2[r,j])*_mp(W1[j,k])*_mp(W1[j,l])*hpp[j] for j in range(neural.H_UNITS)),gp.mpfr(0)) for r in range(3)] for k in range(2) for l in range(2)}
    (P,Q,R),dm,d2m=_metric_ref(ls,dl,d2l)
    F=(-x,-y+x*x);A=((-gp.mpfr(1),gp.mpfr(0)),(2*x,-gp.mpfr(1)))
    md=[sum((dm[l][e]*F[l] for l in range(2)),gp.mpfr(0)) for e in range(3)]
    rate=_mp(neural.RATE)
    C=(md[0]-2*P+4*x*Q+2*rate*P,md[1]-2*Q+2*x*R+2*rate*Q,md[2]-2*R+2*rate*R)
    dc=[]
    for k in range(2):
        dP,dQ,dR=dm[k]
        dmd=[sum((d2m[k,l][e]*F[l]+dm[l][e]*A[l][k] for l in range(2)),gp.mpfr(0)) for e in range(3)]
        dc.append((dmd[0]-2*dP+4*x*dQ+(4*Q if k==0 else 0)+2*rate*dP,
                   dmd[1]-2*dQ+2*x*dR+(2*R if k==0 else 0)+2*rate*dQ,
                   dmd[2]-2*dR+2*rate*dR))
    return C,dc,(P,Q,R)


def _pendulum_ref(theta,state):
    x,y=map(_mp,state)
    ls=[_mp(theta[r,0])+_mp(theta[r,1])*x+_mp(theta[r,2])*y for r in range(3)]
    dl=[[_mp(theta[r,k+1]) for r in range(3)] for k in range(2)]
    (P,Q,R),dm,_=_metric_ref(ls,dl)
    F=(y,-gp.sin(x)-y)
    md=[dm[0][e]*F[0]+dm[1][e]*F[1] for e in range(3)]
    rate=_mp(pendulum.RATE_P);a21=-gp.cos(x)
    return (md[0]+2*a21*Q+2*rate*P,md[1]+a21*R+P-Q+2*rate*Q,md[2]+2*Q-2*R+2*rate*R),(P,Q,R)


def _trig_ref(poly,state):
    x,y=map(_mp,state);s,c=gp.sin(x),gp.cos(x)
    return sum((_mp(v)*x**i*y**j*s**k*c**l for (i,j,k,l),v in poly.coeffs.items()),gp.mpfr(0))


def _lmax(C):
    a,b,c=C
    return (a+c)/2+gp.sqrt(((a-c)/2)**2+b*b)


class BenchmarkRoundoffTests(unittest.TestCase):
    def assertContains(self, interval, reference, index=0):
        lo=float(np.asarray(interval.lo).reshape(-1)[index]);hi=float(np.asarray(interval.hi).reshape(-1)[index])
        self.assertLessEqual(_mp(lo),reference)
        self.assertGreaterEqual(_mp(hi),reference)

    def test_neural_C_and_spatial_derivatives_enclose_high_precision(self):
        rng=np.random.default_rng(73)
        X=np.vstack([rng.uniform(-2,2,(6,2)),np.array([[-2.,2.],[0.,0.],[np.nextafter(0.,1.),-np.nextafter(0.,1.)]])])
        with gp.context(precision=320):
            for seed in (0,3,7):
                parameters=neural.init_params(seed)
                C,dc,M,_=neural._interval_C_and_dC(parameters,Interval(X[:,0]),Interval(X[:,1]))
                grid=neural.grid_enclosure_n(parameters,X,normalized=True)
                for j,state in enumerate(X):
                    ref,dref,mref=_neural_ref(parameters,state)
                    for e in range(3):
                        self.assertContains(C[e],ref[e],j)
                        for k in range(2): self.assertContains(dc[k][e],dref[k][e],j)
                    self.assertContains(grid,_lmax(ref)/(mref[0]+mref[2]),j)

    def test_neural_subbox_bounds_enclose_point_C_and_derivatives(self):
        parameters=neural.init_params(2)
        X=np.array([[-1.75,-.25],[-1.5,0.],[-1.99,-.01]])
        boxx=Interval(np.array([-2.]),np.array([-1.5]));boxy=Interval(np.array([-.5]),np.array([0.]))
        C,dc,_,_=neural._interval_C_and_dC(parameters,boxx,boxy)
        with gp.context(precision=320):
            for state in X:
                ref,dref,_=_neural_ref(parameters,state)
                for e in range(3):
                    self.assertContains(C[e],ref[e])
                    for k in range(2): self.assertContains(dc[k][e],dref[k][e])

    def test_pendulum_exact_coefficients_and_grid_enclose_high_precision(self):
        rng=np.random.default_rng(17)
        X=np.vstack([rng.uniform(-1,1,(8,2)),[[-1.,1.],[1.,-1.],[0.,0.]]])
        with gp.context(precision=320):
            for _ in range(3):
                theta=rng.uniform(-1,1,(3,3))
                polys=pendulum.assemble_p(theta)
                self.assertTrue(all(isinstance(v,Fraction) for p in polys.values() if hasattr(p,'coeffs') for v in p.coeffs.values()))
                grid=pendulum.grid_enclosure_p(theta,X,normalized=True)
                vals=[polys[key].eval_interval(X) for key in ('c11','c12','c22')]
                for j,state in enumerate(X):
                    C,M=_pendulum_ref(theta,state)
                    for e,key in enumerate(('c11','c12','c22')):
                        self.assertContains(vals[e],C[e],j)
                        self.assertLess(abs(_trig_ref(polys[key],state)-C[e]),gp.mpfr('1e-75'))
                    self.assertContains(grid,_lmax(C)/(M[0]+M[2]),j)

    def test_nominal_success_is_rejected_by_neural_outward_check(self):
        with patch.object(neural,'L_norm_n',return_value=0.0), \
             patch.object(neural,'lam_max_C_n',side_effect=lambda p,x,**kw:-np.ones(len(x))), \
             patch.object(neural,'interval_test_n',side_effect=lambda p,x,y,**kw:Interval(-np.ones_like(x.lo),np.full_like(x.lo,1e-16))):
            result=neural.verify_n(neural.init_params(0),3)
        self.assertFalse(result['certified']);self.assertTrue(result['strict_grid_attempted'])
        self.assertFalse(result['grid_enclosed'])
        self.assertGreater(result['worst'],0)

    def test_nominal_success_is_rejected_by_pendulum_outward_check(self):
        theta=np.zeros((3,3))
        with patch.object(pendulum,'L_norm_p',return_value=0.0), \
             patch.object(pendulum,'lam_max_C_p',side_effect=lambda p,x,**kw:-np.ones(len(x))), \
             patch.object(pendulum,'grid_enclosure_p',side_effect=lambda p,x,**kw:Interval(-np.ones(len(x)),np.full(len(x),1e-16))):
            result=pendulum.verify_p(theta,3)
        self.assertFalse(result['certified']);self.assertTrue(result['grid_enclosed'])
        self.assertGreater(result['worst'],0)

    def test_rejection_screen_never_claims_grid_enclosure(self):
        with patch.object(neural,'grid_enclosure_n',side_effect=AssertionError('unexpected expensive pass')):
            result=neural.verify_n(neural.init_params(0),17)
        self.assertFalse(result['certified']);self.assertFalse(result['grid_enclosed'])
        self.assertEqual(result['worst_kind'],'diagnostic_rejection')

    def test_large_batch_neural_grid_encloses_high_precision_reference(self):
        X=np.random.default_rng(21).uniform(-2,2,(4_101,2))
        parameters=neural.init_params(3)
        values=neural.grid_enclosure_n(parameters,X,normalized=True)
        with gp.context(precision=320):
            for j in (0,17,1024,4095,4100):
                C,_,M=_neural_ref(parameters,X[j])
                self.assertContains(values,_lmax(C)/(M[0]+M[2]),j)


class ProducerContractTests(unittest.TestCase):
    def test_producers_persist_rigorous_flags_and_cached_example_without_reverification(self):
        import matplotlib.pyplot as plt
        import run_all
        from rounding import grid_radius,mul_up,sub_down
        tau=grid_radius(np.linspace(-2,2,3))
        margin=-mul_up(0.,tau)
        V={"certified":True,"L":0.,"tau":tau,"margin":margin,"worst":-1.,
           "uniform_slack":sub_down(1.,-margin),"grid_enclosed":True,"outward_rounded":True,
           "rounding_mode":"outward","bounds_s":0.,"viol":np.zeros((0,2)),
           "nominal_worst":-1.,"gridpoints_covered":9,"outward_box_evaluations":1,
           "outward_grid_evaluations":0,"nominal_grid_evaluations":9,"oracle_evaluations":10,
           "grid_check_method":"disjoint_index_rectangles"}
        V.update(neural.certified_metric_outputs_n(neural.init_params(0),V["uniform_slack"]))
        keys=("worst","L","tau","grid_enclosed","outward_rounded","uniform_slack","rounding_mode","gridpoints_covered")
        trace=[dict({k:V[k] for k in keys},iter=0,gap=-1.)]
        with tempfile.TemporaryDirectory() as folder,patch.object(plt,'savefig'),contextlib.redirect_stdout(io.StringIO()):
            with patch.object(neural,'__file__',str(Path(folder)/'bench_neural.py')), \
                 patch.object(neural,'run_neural',return_value=(neural.init_params(0),trace,V,True)), \
                 patch.object(neural,'verify_n',return_value=V) as verification:
                output=neural.experiment(n_seeds=1,op_grid=3,coarse_grid=3,validation={"mock_checks":True})
            # One coarse postcheck, no duplicate representative-grid check.
            self.assertEqual(verification.call_count,1)
            run_all.validate_benchmark(output,neural=True)
            self.assertEqual(output["config"]["max_rounds"],neural.MAX_ROUNDS)
            self.assertEqual(output["config"]["training_steps"],neural.TRAINING_STEPS)
            self.assertEqual(output["example"]["b"],output["runs"]["0"]["b"])
            self.assertTrue(output["example"]["metric_bounds_outward_rounded"])
            self.assertTrue((Path(folder)/'results/neural.json').is_file())
            with patch.object(pendulum,'__file__',str(Path(folder)/'bench_pendulum.py')), \
                 patch.object(pendulum,'run_pendulum',return_value=(np.zeros((3,3)),trace,V,True)), \
                 patch.object(pendulum,'verify_p',side_effect=AssertionError('duplicate example verification')):
                output=pendulum.experiment(n_seeds=1,n_grid=3,validation={"mock_checks":True})
            run_all.validate_benchmark(output)
            self.assertTrue((Path(folder)/'results/pendulum.json').is_file())


class GroupedGridProofTests(unittest.TestCase):
    def test_whole_rectangle_proves_exact_logical_grid_coverage(self):
        nodes=np.linspace(-2,2,11)
        with patch.object(neural,'interval_test_n',side_effect=lambda p,x,y,**kw:Interval(np.full_like(x.lo,-2.0))):
            result=neural._rounded_grid_boxes(None,nodes,-1.0)
        self.assertTrue(result['complete']);self.assertTrue(result['certified'])
        self.assertEqual(result['boxes_evaluated'],1)
        self.assertEqual(result['gridpoints_covered'],121)
        self.assertEqual(result['worst'],-2.0)

    def test_disjoint_uneven_splits_and_unattainable_guide_use_original_node_margin(self):
        nodes=np.linspace(-1,1,7)
        visited=[]
        def enclosure(p,x,y,**kw):
            # True function is the constant -0.3. A deliberately dependent
            # box enclosure forces every node to be checked separately.
            widths=(x.hi-x.lo)+(y.hi-y.lo)
            single=(x.lo==x.hi)&(y.lo==y.hi)
            visited.extend(zip(x.lo[single].tolist(),y.lo[single].tolist()))
            return Interval(-.3-widths,-.3+widths)
        with patch.object(neural,'interval_test_n',side_effect=enclosure):
            result=neural._rounded_grid_boxes(None,nodes,-.1,target=-1.0)
        self.assertTrue(result['complete']);self.assertTrue(result['certified'])
        self.assertEqual(result['gridpoints_covered'],49)
        self.assertEqual(len(visited),49);self.assertEqual(len(set(visited)),49)
        self.assertEqual(set(visited),set((x,y) for x in nodes for y in nodes))
        self.assertLessEqual(result['worst'],-.1)

    def test_isolated_corner_violation_cannot_certify(self):
        nodes=np.linspace(0,1,7)
        with patch.object(neural,'interval_test_n',side_effect=lambda p,x,y,**kw:x+y):
            result=neural._rounded_grid_boxes(None,nodes,1.75)
        self.assertFalse(result['complete']);self.assertFalse(result['certified'])
        self.assertFalse(result['fallback'])
        self.assertGreater(result['worst'],1.75)
        self.assertLess(result['gridpoints_covered'],49)

    def test_cap_requests_full_outward_fallback_never_partial_positive(self):
        nodes=np.linspace(-1,1,5)
        with patch.object(neural,'interval_test_n',side_effect=lambda p,x,y,**kw:Interval(-np.ones_like(x.lo),np.ones_like(x.lo))):
            result=neural._rounded_grid_boxes(None,nodes,-.1,max_evaluations=1)
        self.assertFalse(result['complete']);self.assertTrue(result['fallback'])
        self.assertEqual(result['gridpoints_covered'],0)
        proof={"complete":False,"fallback":True,"boxes_evaluated":1,"gridpoints_covered":0}
        with patch.object(neural,'L_norm_n',return_value=0.0), \
             patch.object(neural,'lam_max_C_n',side_effect=lambda p,x,**kw:-np.ones(len(x))), \
             patch.object(neural,'_rounded_grid_boxes',return_value=proof), \
             patch.object(neural,'grid_enclosure_n',side_effect=lambda p,x,**kw:Interval(-np.ones(len(x)))):
            verified=neural.verify_n(neural.init_params(0),5)
        self.assertTrue(verified['certified']);self.assertTrue(verified['grid_enclosed'])
        self.assertEqual(verified['gridpoints_covered'],25)
        self.assertEqual(verified['outward_grid_evaluations'],25)
        self.assertEqual(verified['grid_check_method'],'streaming_point_intervals')


if __name__=='__main__': unittest.main()
