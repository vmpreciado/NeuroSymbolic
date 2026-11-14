"""Independent mathematical and tamper checks for the physical-model oracle."""
import copy
from fractions import Fraction
import itertools
import unittest

import autograd
import gmpy2
import numpy as np

import application_verifier as app
import bench_neural
from rounding import Interval as Iv


def constant_factor():
    # At the equilibrium, A.T P+P A=-I for P=[[2,1],[1,3]]. The stored
    # factor differs by roundoff, which the verifier accounts for directly.
    p = np.zeros(39)
    c = np.sqrt(2.9)
    b = 1/c
    a = np.sqrt(1.9-b*b)
    p[-3:] = [a, b, c]
    return p


def mpfr_contraction(theta, state, rate=.05, alpha=.1):
    """Independent 240-bit scalar evaluation with the original shifted sine."""
    mp = gmpy2.mpfr
    with gmpy2.context(precision=240):
        p, x = [mp(float(v)) for v in theta], [mp(float(v)) for v in state]
        w1 = [p[2*k:2*k+2] for k in range(6)]
        b1 = p[12:18]
        w2 = [p[18+6*k:24+6*k] for k in range(3)]
        b2 = p[36:39]
        h = [gmpy2.tanh(w[0]*x[0]+w[1]*x[1]+b) for w, b in zip(w1,b1)]
        hp = [1-v*v for v in h]
        l = [sum((w*h0 for w,h0 in zip(row,h)),mp(0))+bias for row,bias in zip(w2,b2)]
        dl = [[sum((w2[r][j]*hp[j]*w1[j][k] for j in range(6)),mp(0)) for r in range(3)] for k in range(2)]
        a,b,c = l
        P,Q,R = mp(alpha)+a*a+b*b,b*c,mp(alpha)+c*c
        dM = [(2*a*da+2*b*db,db*c+b*dc,2*c*dc) for da,db,dc in dl]
        angle = gmpy2.const_pi()/3
        f = [x[1], -mp(.5)*x[1] - gmpy2.sin(x[0]+angle)+gmpy2.sin(angle)]
        k = -gmpy2.cos(x[0]+angle)
        md = [dM[0][j]*f[0]+dM[1][j]*f[1] for j in range(3)]
        return [md[0]+2*k*Q+2*mp(rate)*P,
                md[1]+P-mp(.5)*Q+k*R+2*mp(rate)*Q,
                md[2]+2*Q-R+2*mp(rate)*R]


class SwingVerifierTests(unittest.TestCase):
    def test_interval_covers_high_precision_original_shifted_sine(self):
        rng = np.random.default_rng(912)
        for seed in range(3):
            theta = bench_neural.init_params(seed)
            for _ in range(4):
                center = rng.uniform(-.7,.7,2)
                radius = rng.uniform(.001,.04,2)
                lower,upper = center-radius,center+radius
                matrix = app.contraction_interval(theta,lower,upper)
                intervals = [matrix[0][0],matrix[0][1],matrix[1][1]]
                for state in (center,lower,upper):
                    values = mpfr_contraction(theta,state)
                    for interval,value in zip(intervals,values):
                        self.assertLessEqual(gmpy2.mpfr(float(interval.lo)),value)
                        self.assertGreaterEqual(gmpy2.mpfr(float(interval.hi)),value)

    def test_nominal_entries_match_independent_high_precision(self):
        theta = bench_neural.init_params(29)
        points = np.array([[0.,0.],[.2,-.3],[-.4,.5]])
        entries = app.contraction_entries(theta,points)
        for row,state in enumerate(points):
            np.testing.assert_allclose([a[row] for a in entries[:3]],
                                       [float(v) for v in mpfr_contraction(theta,state)],rtol=2e-13,atol=2e-13)
        grad = autograd.grad(lambda p: app.contraction_nominal(p,points,normalized=True).sum())(theta)
        self.assertTrue(np.isfinite(grad).all())

    def test_batch_interval_matches_scalar_and_hull_corners(self):
        theta = bench_neural.init_params(31)
        lower = np.array([[-.3,-.1],[.1,.2]])
        upper = lower+.01
        batch = app._contraction_batch(theta,lower,upper)
        for row in range(2):
            scalar = app.contraction_interval(theta,lower[row],upper[row])
            for b,s in zip(batch,(scalar[0][0],scalar[0][1],scalar[1][1])):
                self.assertEqual(b.lo[row],s.lo)
                self.assertEqual(b.hi[row],s.hi)
        c11,c12,c22 = Iv(-3.,-2.),Iv(-.5,.6),Iv(-2.,-1.)
        bound = float(app._hull_eigen_upper(c11,c12,c22))
        with gmpy2.context(precision=240):
            for p,q,r in itertools.product([-3.,-2.],[-.5,.6],[-2.,-1.]):
                p,q,r = map(gmpy2.mpfr,(p,q,r))
                eigen=(p+r)/2+gmpy2.sqrt(((p-r)/2)**2+q*q)
                self.assertLessEqual(eigen,gmpy2.mpfr(bound))

    def test_complete_partition_and_tamper_rejection(self):
        theta=constant_factor()
        result=app.adaptive_oracle(theta,[-.125]*2,[.125]*2,max_boxes=8191)
        self.assertTrue(result['certified'])
        self.assertGreater(app.replay_oracle_certificate(result,theta),1)
        broken=copy.deepcopy(result)
        broken['proof']['leaf_paths'].pop()
        with self.assertRaises(ValueError):app.replay_oracle_certificate(broken,theta)
        broken=copy.deepcopy(result)
        broken['target_margin']=10.
        broken['uniform_raw_margin_lower']=10.
        with self.assertRaises(ValueError):app.replay_oracle_certificate(broken,theta)
        changed=theta.copy();changed[-1]+=1e-5
        with self.assertRaises(ValueError):app.replay_oracle_certificate(result,changed)

    def test_failure_and_budget_are_inconclusive(self):
        theta=bench_neural.init_params(0)
        result=app.adaptive_oracle(theta,[-.5]*2,[.5]*2,max_boxes=1)
        self.assertFalse(result['certified'])
        self.assertEqual(result['reason'],'box_budget_exhausted')
        self.assertIsNone(result['uniform_raw_margin_lower'])
        self.assertEqual(result['proof']['leaf_paths'],[])
        self.assertEqual(app.replay_oracle_certificate(result,theta),0)
        result=app.adaptive_oracle(theta,[-.5]*2,[.5]*2,check_points=np.array([[0.,0.]]))
        self.assertFalse(result['certified'])
        self.assertEqual(result['reason'],'nominal_screen_inconclusive')
        self.assertEqual(result['counters']['boxes_evaluated'],0)

    def test_metric_rate_bounds_are_outward(self):
        theta=bench_neural.init_params(12)
        result=app.metric_guarantees(theta,[-.5]*2,[.5]*2,.05,1e-4)
        F=Fraction.from_float
        self.assertLessEqual(F(result['rate_lower']),F(.05)+F(1e-4)/(2*F(result['b'])))
        self.assertGreaterEqual(F(result['overshoot_upper'])**2,F(result['b'])/F(result['a']))
        X=np.random.default_rng(112).uniform(-.5,.5,(100,2))
        (a,b,c),_=bench_neural.L_entries(theta,X)
        metric=np.array([[[.1+a0*a0+b0*b0,b0*c0],[b0*c0,.1+c0*c0]] for a0,b0,c0 in zip(a,b,c)])
        self.assertTrue(np.all(np.linalg.eigvalsh(metric)[:,-1]<=result['b']))

    def test_energy_level_below_exact_boundary(self):
        for B in (.25,.5,.75):
            report=app.certified_energy_level([-B]*2,[B]*2)
            with gmpy2.context(precision=240):
                x=gmpy2.mpfr(B)
                r=gmpy2.sqrt(3)/2
                left=(1-gmpy2.cos(-x))/2+r*(gmpy2.sin(-x)+x)
                right=(1-gmpy2.cos(x))/2+r*(gmpy2.sin(x)-x)
                boundary=min(left,right,x*x/2)
                self.assertGreater(report['rho_lower'],0)
                self.assertLessEqual(gmpy2.mpfr(report['boundary_energy_lower']),boundary)
                self.assertLessEqual(gmpy2.mpfr(report['rho_lower']),gmpy2.mpfr('0.9')*boundary)
            self.assertEqual(report['convex'],B<=.5)
        with self.assertRaises(ValueError):app.certified_energy_level([-.8]*2,[.8]*2)


if __name__=='__main__':unittest.main()
