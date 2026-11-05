"""bench_pendulum.py (E3): a non-polynomial benchmark and its certified oracle.

Benchmark (trigonometric dynamics, realizability not assumed):
    f(x) = ( x2,  -sin(x1) - x2 )   on   X = [-1, 1]^2,
a damped pendulum. The metric class is unchanged from the polynomial benchmark,
M_theta = alpha I + L(x)^T L(x) with L linear in (1, x1, x2), so M and its spatial
derivatives are polynomials; only the dynamics are non-polynomial.

Certified oracle. The contraction matrix C and its partials are polynomials in
the four atoms (x1, x2, s, c) with s = sin(x1), c = cos(x1). We represent them in
a small class TrigPoly closed under +, *, and the derivations
    d/dx1: x1 -> 1, s -> c, c -> -s, x2 -> 0,
    d/dx2: x2 -> 1, x1, s, c -> 0.
A sound sup bound over a box treats s and c as bounded atoms, |s|, |c| <= 1,
tightened per subbox by |sin x1| <= |sin c1| + h and |cos x1| <= |cos c1| + h
(1-Lipschitz), and bounds x1, x2 by their subbox ranges. Subdivision tightens the
bound. This is interval bound propagation with sin, cos as bounded atoms; it is
sound in real arithmetic and needs no approximation remainder terms. The
implementation retains exact rational coefficients through assembly and uses
outward-rounded interval bounds, transcendental enclosures, and grid tests.
Stored binary64 parameters define the exact real-valued candidate.
"""
import math
import time
from fractions import Fraction
import autograd.numpy as np
import numpy as onp

import rounding
from rounding import Interval as Iv, add_up, mul_up, div_up, sqrt_up, sub_down, float_down, float_up

import cegis_c
from cegis_c import feats, DPHI, ALPHA

BOX_P = 1.0     # domain [-1, 1]^2
RATE_P = 0.2    # target rate for this experiment; no infeasibility claim at other rates
MAX_ROUNDS = 10
GROWTH_FACTOR = 100.0
TRAINING_STEPS = 500
LEARNING_RATE = 2e-2
INITIAL_SAMPLES = 500
BASE_MARGIN = 0.05
GAMMA = 1.3
K_WORST = 60
PERTURBATIONS = 5
PERTURBATION_STD = 0.05


# ----------------------- trig-polynomial class -----------------------

def _q(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite trigonometric polynomial coefficient")
    return Fraction.from_float(value)


class TrigPoly:
    """sum coeff * x1^i x2^j s^k c^l, with s = sin(x1), c = cos(x1).
    Key is (i, j, k, l); zero coefficients pruned."""

    __slots__ = ("coeffs",)

    def __init__(self, coeffs=None):
        self.coeffs = {}
        if coeffs:
            for key, v in coeffs.items():
                if v != 0.0:
                    self.coeffs[key] = _q(v)

    @classmethod
    def const(cls, a):
        return cls({(0, 0, 0, 0): a}) if a != 0.0 else cls()

    @classmethod
    def atom(cls, which):
        return cls({{"x1": (1, 0, 0, 0), "x2": (0, 1, 0, 0),
                     "s": (0, 0, 1, 0), "c": (0, 0, 0, 1)}[which]: 1.0})

    def _add(self, key, v):
        nv = self.coeffs.get(key, Fraction(0)) + _q(v)
        if nv == 0.0:
            self.coeffs.pop(key, None)
        else:
            self.coeffs[key] = nv

    def __add__(self, o):
        if isinstance(o, (int, float, Fraction)):
            o = TrigPoly.const(o)
        out = TrigPoly(self.coeffs)
        for k, v in o.coeffs.items():
            out._add(k, v)
        return out
    __radd__ = __add__

    def __neg__(self):
        return TrigPoly({k: -v for k, v in self.coeffs.items()})

    def __sub__(self, o):
        return self + (-o if isinstance(o, TrigPoly) else -_q(o))
    __rsub__ = lambda self, o: (-self) + o

    def __mul__(self, o):
        if isinstance(o, (int, float, Fraction)):
            if o == 0.0:
                return TrigPoly()
            return TrigPoly({k: v * _q(o) for k, v in self.coeffs.items()})
        out = TrigPoly()
        for k1, v1 in self.coeffs.items():
            for k2, v2 in o.coeffs.items():
                out._add(tuple(a + b for a, b in zip(k1, k2)), v1 * v2)
        return out
    __rmul__ = __mul__

    def deriv(self, var):
        """Partial derivative. var=0 -> d/dx1 (x1->1, s->c, c->-s); var=1 -> d/dx2."""
        out = TrigPoly()
        if var == 1:
            for (i, j, k, l), v in self.coeffs.items():
                if j >= 1:
                    out._add((i, j - 1, k, l), j * v)
            return out
        for (i, j, k, l), v in self.coeffs.items():
            if i >= 1:
                out._add((i - 1, j, k, l), i * v)          # d x1^i
            if k >= 1:
                out._add((i, j, k - 1, l + 1), k * v)      # d s^k = k s^{k-1} c
            if l >= 1:
                out._add((i, j, k + 1, l - 1), -l * v)     # d c^l = -l c^{l-1} s
        return out

    def evalv(self, X):
        """Numeric evaluation on X shape (N,2), using true sin/cos of x1."""
        x1, x2 = X[:, 0], X[:, 1]
        s, c = np.sin(x1), np.cos(x1)
        out = np.zeros(X.shape[0])
        for (i, j, k, l), v in self.coeffs.items():
            out = out + float(v) * x1**i * x2**j * s**k * c**l
        return out

    def eval_interval(self, X):
        """Outward value enclosure, including exact rational coefficients."""
        x1, x2 = Iv(X[:, 0]), Iv(X[:, 1])
        sine, cosine = x1.sin(), x1.cos()
        out = Iv(onp.zeros(len(X)))
        for (i, j, k, l), value in self.coeffs.items():
            coeff = Iv(float_down(value), float_up(value))
            out = out + coeff * x1**i * x2**j * sine**k * cosine**l
        return out

    def center_fraction(self, c1, c2):
        """Exact value for a polynomial independent of sine/cosine atoms."""
        if any(k or l for i, j, k, l in self.coeffs):
            raise ValueError("center_fraction requires a polynomial in the state")
        return sum((v*_q(c1)**i*_q(c2)**j for (i,j,k,l),v in self.coeffs.items()), Fraction(0))

    def sup_subbox(self, c1, c2, h):
        """Upper bound with exact coefficients and directed transcendental radii."""
        c1q, c2q, hq = _q(c1), _q(c2), _q(h)
        rx1, rx2 = abs(c1q) + hq, abs(c2q) + hq
        # Binary64 dyadic subbox centers are exact; MPFR supplies sin/cos bounds.
        if Fraction.from_float(float(c1q)) != c1q:
            center = Iv(float_down(c1q), float_up(c1q))
        else:
            center = Iv(float(c1q))
        rs = min(Fraction(1), _q(float(center.sin().absmax())) + hq)
        rc = min(Fraction(1), _q(float(center.cos().absmax())) + hq)
        total = sum((abs(v) * rx1**i * rx2**j * rs**k * rc**l
                     for (i,j,k,l),v in self.coeffs.items()), Fraction(0))
        return float_up(total)

    def supabs_subdiv(self, B, depth):
        """Max over 4^depth subboxes of sup_subbox. Sound; tighter with depth."""
        if depth <= 0:
            return self.sup_subbox(0.0, 0.0, B)
        n = 2 ** depth
        h = B / n
        best = 0.0
        for i in range(n):
            c1 = -B + (2 * i + 1) * h
            for j in range(n):
                c2 = -B + (2 * j + 1) * h
                b = self.sup_subbox(c1, c2, h)
                if b > best:
                    best = b
        return best


def _from_poly2(p):
    """Convert a bounds.Poly2 (keys (i,j)) to a TrigPoly (keys (i,j,0,0))."""
    return TrigPoly({(i, j, 0, 0): v for (i, j), v in p.coeffs.items()})


# ----------------------- metric parts (polynomial, f-independent) -----------------------

def _metric_numeric(theta, X):
    """M entries p,q,r and spatial derivatives dm[k] = (dp,dq,dr) wrt x_k.
    Metric-only (no dynamics); shared with the polynomial benchmark's class."""
    PH = feats(X)
    l11 = PH @ theta[0]; l21 = PH @ theta[1]; l22 = PH @ theta[2]
    p = ALPHA + l11**2 + l21**2
    q = l21 * l22
    r = ALPHA + l22**2
    dm = []
    for k in range(2):
        dl11 = theta[0] @ DPHI[k]; dl21 = theta[1] @ DPHI[k]; dl22 = theta[2] @ DPHI[k]
        dp = 2 * l11 * dl11 + 2 * l21 * dl21
        dq = dl21 * l22 + l21 * dl22
        dr = 2 * l22 * dl22
        dm.append((dp, dq, dr))
    return p, q, r, dm


# ----------------------- numeric contraction matrix for the pendulum -----------------------

def lam_max_C_p(theta, X, normalized=False):
    """lambda_max of C for the pendulum f=(x2,-sin x1 - x2), metric M_theta.
    A = [[0,1],[-cos x1,-1]]; Mdot = dM/dx1 * x2 + dM/dx2 * (-sin x1 - x2)."""
    p, q, r, dm = _metric_numeric(theta, X)
    x1 = X[:, 0]; x2 = X[:, 1]
    f1 = x2
    f2 = -np.sin(x1) - x2
    dp = dm[0][0] * f1 + dm[1][0] * f2
    dq = dm[0][1] * f1 + dm[1][1] * f2
    dr = dm[0][2] * f1 + dm[1][2] * f2
    a11, a12, a21, a22 = 0.0, 1.0, -np.cos(x1), -1.0
    s11 = 2 * (a11 * p + a21 * q)
    s22 = 2 * (a12 * q + a22 * r)
    s12 = a11 * q + a21 * r + a12 * p + a22 * q
    c11 = dp + s11 + 2 * RATE_P * p
    c22 = dr + s22 + 2 * RATE_P * r
    c12 = dq + s12 + 2 * RATE_P * q
    m = 0.5 * (c11 + c22)
    d = 0.5 * (c11 - c22)
    lam = m + np.sqrt(d**2 + c12**2 + 1e-300)
    return lam / (p + r) if normalized else lam


# ----------------------- symbolic (TrigPoly) assembly -----------------------

def assemble_p(theta):
    """C entries and their partials as TrigPoly, for the pendulum."""
    import bounds
    P = bounds.assemble_exact(theta)    # exact metric coefficients before any rounding
    m11 = _from_poly2(P["m11"]); m12 = _from_poly2(P["m12"]); m22 = _from_poly2(P["m22"])
    dm = [tuple(_from_poly2(e) for e in P["dm"][k]) for k in range(2)]
    x2 = TrigPoly.atom("x2")
    s = TrigPoly.atom("s"); c = TrigPoly.atom("c")
    f1 = x2
    f2 = -s - x2
    f = [f1, f2]
    mdot11 = dm[0][0] * f[0] + dm[1][0] * f[1]
    mdot12 = dm[0][1] * f[0] + dm[1][1] * f[1]
    mdot22 = dm[0][2] * f[0] + dm[1][2] * f[1]
    a11 = TrigPoly.const(0.0); a12 = TrigPoly.const(1.0)
    a21 = -c; a22 = TrigPoly.const(-1.0)
    s11 = a11 * m11 * 2.0 + a21 * m12 * 2.0
    s22 = a22 * m22 * 2.0 + a12 * m12 * 2.0
    s12 = a11 * m12 + a21 * m22 + a12 * m11 + a22 * m12
    c11 = mdot11 + s11 + m11 * (_q(RATE_P) * 2)
    c22 = mdot22 + s22 + m22 * (_q(RATE_P) * 2)
    c12 = mdot12 + s12 + m12 * (_q(RATE_P) * 2)
    dc = [(c11.deriv(k), c12.deriv(k), c22.deriv(k)) for k in range(2)]
    v = m11 + m22
    dv = [v.deriv(0), v.deriv(1)]
    return {"c11": c11, "c12": c12, "c22": c22, "dc": dc, "v": v, "dv": dv,
            "m11": m11, "m12": m12, "m22": m22}


def _sb(p, depth):
    return p.supabs_subdiv(BOX_P, depth)


def L_raw_p(theta, depth=3, P=None):
    P = assemble_p(theta) if P is None else P
    total = Iv(0.0)
    for d11,d12,d22 in P["dc"]:
        total = total + Iv(_sb(d11,depth)).square() + Iv(_sb(d12,depth)).square()*2.0 + Iv(_sb(d22,depth)).square()
    return float(total.sqrt().hi)


def _subboxes(depth):
    n = 2 ** depth
    h = Fraction.from_float(BOX_P) / n
    return [(-_q(BOX_P) + (2*i+1)*h, -_q(BOX_P) + (2*j+1)*h)
            for i in range(n) for j in range(n)], h


def _gtsup_and_vmin(P, depth):
    pF2 = P["c11"]*P["c11"] + P["c12"]*P["c12"]*2 + P["c22"]*P["c22"]
    v = P["v"]
    centers,h = _subboxes(depth)
    gt_sup = 0.0; v_min = float("inf")
    floor = _q(ALPHA)*2
    for c1,c2 in centers:
        vc = v.center_fraction(c1,c2)
        deviation = _q((v-TrigPoly.const(vc)).sup_subbox(c1,c2,h))
        inf_v = float_down(max(floor, vc-deviation))
        v_min = min(v_min,inf_v)
        ratio = div_up(sqrt_up(pF2.sup_subbox(c1,c2,h)),inf_v)
        gt_sup = max(gt_sup,ratio)
    return gt_sup,max(float_down(floor),v_min)


def L_norm_p(theta, depth=3):
    P = assemble_p(theta)
    Lg = L_raw_p(theta,depth,P=P)
    Lv = float((Iv(_sb(P["dv"][0],depth)).square()+Iv(_sb(P["dv"][1],depth)).square()).sqrt().hi)
    gt_sup,v_min = _gtsup_and_vmin(P,depth)
    return div_up(add_up(Lg,mul_up(gt_sup,Lv)),v_min)


def grid_enclosure_p(theta, X, normalized=True):
    """Outward spectral test from the original parameter values at grid points."""
    x1,x2 = Iv(X[:,0]),Iv(X[:,1])
    ls = [Iv(float(theta[r,0])) + x1*float(theta[r,1]) + x2*float(theta[r,2]) for r in range(3)]
    l11,l21,l22 = ls
    P = l11.square()+l21.square()+ALPHA
    Q = l21*l22
    R = l22.square()+ALPHA
    dM = []
    for k in range(2):
        a,b,c = [float(theta[r,k+1]) for r in range(3)]
        dM.append((l11*a*2.0+l21*b*2.0, l22*b+l21*c, l22*c*2.0))
    f1,f2 = x2,-x1.sin()-x2
    md = [dM[0][e]*f1+dM[1][e]*f2 for e in range(3)]
    a21 = -x1.cos()
    c11 = md[0]+a21*Q*2.0+P*Iv(RATE_P)*2.0
    c12 = md[1]+a21*R+P-Q+Q*Iv(RATE_P)*2.0
    c22 = md[2]+(Q-R)*2.0+R*Iv(RATE_P)*2.0
    result = rounding.lambda_max_2x2(c11,c12,c22)
    return result/(P+R) if normalized else result


# ----------------------- verifier -----------------------

def grid_p(n):
    s = np.linspace(-BOX_P, BOX_P, n)
    XX, YY = np.meshgrid(s, s)
    return np.stack([XX.ravel(), YY.ravel()], axis=1), s


def verify_p(theta, n_grid, normalized=True, depth=3):
    """Reject cheaply with nominal values; certify only with outward grid bounds."""
    if n_grid < 2:
        raise ValueError("n_grid must be >= 2")
    X,s = grid_p(n_grid)
    if s[0] != -BOX_P or s[-1] != BOX_P:
        raise ValueError("grid does not include the exact domain endpoints")
    tau = rounding.grid_radius(s,dimension=2)
    t_bounds = time.perf_counter()
    L = L_norm_p(theta,depth) if normalized else L_raw_p(theta,depth)
    bounds_s = time.perf_counter()-t_bounds
    if not math.isfinite(L) or L<0:
        raise ValueError("invalid Lipschitz bound")
    g = onp.asarray(lam_max_C_p(theta,X,normalized=normalized))
    if not onp.isfinite(g).all():
        raise ValueError("nonfinite pendulum test value")
    margin = -mul_up(L,tau)
    viol = g>margin
    grid_enclosed = not viol.any()
    worst = float(g.max())
    if grid_enclosed:
        upper = onp.empty(len(X))
        for start in range(0,len(X),65_536):
            upper[start:start+65_536] = grid_enclosure_p(theta,X[start:start+65_536],normalized=normalized).hi
        if not onp.isfinite(upper).all():
            raise ValueError("nonfinite outward pendulum grid bound")
        worst = float(upper.max())
        viol = upper>margin
        if viol.any():
            g = upper  # Inconclusive points guide the next heuristic learner step.
    uniform_slack = sub_down(-worst,-margin)
    certified = grid_enclosed and not viol.any() and uniform_slack>=0.0
    if not certified and not viol.any():
        viol[0] = True
        worst = float(onp.nextafter(margin,onp.inf))
        uniform_slack = sub_down(-worst,-margin)
    return {"certified":bool(certified),"X":X,"g":g,"viol":viol,
            "margin":margin,"L":L,"tau":tau,"worst":worst,"bounds_s":bounds_s,
            "grid_enclosed":bool(grid_enclosed),"rounding_mode":"outward","outward_rounded":True,
            "worst_kind":"outward_grid_upper" if grid_enclosed else "diagnostic_rejection",
            "uniform_slack":uniform_slack,
            "nominal_grid_evaluations":n_grid**2,
            "outward_grid_evaluations":n_grid**2 if grid_enclosed else 0,
            "oracle_evaluations":n_grid**2*(1+int(grid_enclosed)),
            "gridpoints_covered":n_grid**2 if grid_enclosed else 0}


# ----------------------- self-tests (soundness firewall) -----------------------

def test_assembly(seed=0, n=200, tol=1e-9):
    """TrigPoly C entries match the numeric assembly at random (theta, x)."""
    rng = onp.random.default_rng(seed)
    maxrel = 0.0
    for _ in range(n):
        theta = rng.uniform(-1, 1, size=(3, 3))
        X = rng.uniform(-BOX_P, BOX_P, size=(1, 2))
        P = assemble_p(theta)
        # numeric c11,c12,c22
        p, q, r, dm = _metric_numeric(theta, X)
        x1 = X[:, 0]; x2 = X[:, 1]
        f1 = x2; f2 = -np.sin(x1) - x2
        dp = dm[0][0]*f1 + dm[1][0]*f2; dq = dm[0][1]*f1 + dm[1][1]*f2; dr = dm[0][2]*f1 + dm[1][2]*f2
        a21 = -np.cos(x1)
        s11 = 2*(a21*q); s22 = 2*(1.0*q + (-1.0)*r); s12 = a21*r + 1.0*p + (-1.0)*q
        c11 = dp + s11 + 2*RATE_P*p; c22 = dr + s22 + 2*RATE_P*r; c12 = dq + s12 + 2*RATE_P*q
        for sym, num in [(P["c11"], c11), (P["c12"], c12), (P["c22"], c22)]:
            rel = abs(sym.evalv(X)[0] - num[0]) / max(abs(num[0]), 1.0)
            maxrel = max(maxrel, rel)
    print(f"[pendulum assembly] {n} pairs, max rel err = {maxrel:.2e}")
    return maxrel < tol


def test_enclosure_sound(seed=1, n_theta=8, n_grid=401, depth=3):
    """Certified sup bounds enclose the true sup on a dense grid, and the
    certified normalized Lipschitz bound exceeds the empirical estimate."""
    rng = onp.random.default_rng(seed)
    fails = 0
    ratios = []
    X, _ = grid_p(n_grid)
    for _ in range(n_theta):
        theta = rng.uniform(-1, 1, size=(3, 3))
        P = assemble_p(theta)
        # sup soundness: |c11| bound >= dense max
        for key in ("c11", "c12", "c22"):
            dense = float(np.max(np.abs(P[key].evalv(X))))
            cert = _sb(P[key], depth)
            if cert + 1e-9 < dense:
                fails += 1
        # Lipschitz soundness: certified L_norm >= finite-difference estimate
        Lc = L_norm_p(theta, depth)
        hstep = 1e-5
        g0 = lam_max_C_p(theta, X, normalized=True)
        gx = (lam_max_C_p(theta, X + np.array([hstep, 0.0]), normalized=True) - g0) / hstep
        gy = (lam_max_C_p(theta, X + np.array([0.0, hstep]), normalized=True) - g0) / hstep
        Lemp = float(np.max(np.sqrt(gx**2 + gy**2)))
        ratios.append(Lc / Lemp if Lemp > 0 else float("inf"))
        if Lc + 1e-9 < Lemp:
            fails += 1
    print(f"[pendulum enclosure] fails={fails}, cert/emp L ratio "
          f"min={min(ratios):.2f} median={np.median(ratios):.2f} max={max(ratios):.2f}")
    return fails == 0


# ----------------------- learner and experiment -----------------------

from autograd import grad as _grad
import numpy.random as npr


def _loss(theta, X, eps):
    return np.mean(np.maximum(0.0, lam_max_C_p(theta, X, normalized=True) + eps))


_dloss = _grad(_loss)


def _adam(theta, X, eps, steps=TRAINING_STEPS, lr=LEARNING_RATE):
    m = onp.zeros_like(theta); v = onp.zeros_like(theta)
    b1, b2, e = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        g = _dloss(theta, X, eps)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        theta = theta - lr * (m / (1 - b1 ** t)) / (onp.sqrt(v / (1 - b2 ** t)) + e)
    return theta


def run_pendulum(seed=0, n_grid=401, max_iter=MAX_ROUNDS, gamma=GAMMA, depth=3, timings=None):
    """CEGIS-C on the pendulum, normalized coupled variant. Returns
    (theta, trace, V, certified)."""
    npr.seed(seed)
    D = npr.uniform(-BOX_P, BOX_P, size=(INITIAL_SAMPLES, 2))
    theta = 0.1 * npr.randn(3, 3); theta[0, 0] = 1.0; theta[2, 0] = 1.0
    eps = BASE_MARGIN
    trace = []
    train_s = 0.0
    for it in range(max_iter + 1):
        t_verify = time.perf_counter()
        V = verify_p(theta, n_grid, normalized=True, depth=depth)
        verify_s = time.perf_counter() - t_verify
        if timings is not None:
            timings.append({"iter": it, "train_s": train_s,
                            "verify_including_bounds_s": verify_s,
                            "bounds_s": V["bounds_s"],
                            "oracle_evaluations": V["oracle_evaluations"],
                            "nominal_grid_evaluations": V["nominal_grid_evaluations"],
                            "outward_grid_evaluations": V["outward_grid_evaluations"],
                            "gridpoints_covered": V["gridpoints_covered"]})
        trace.append({"iter": it, "worst": V["worst"], "L": V["L"],
                      "gap": V["worst"] - V["margin"],
                      "oracle_evaluations": V["oracle_evaluations"],
                      "nominal_grid_evaluations": V["nominal_grid_evaluations"],
                      "outward_grid_evaluations": V["outward_grid_evaluations"],
                      "gridpoints_covered": V["gridpoints_covered"],
                      "grid_enclosed": V["grid_enclosed"], "rounding_mode": "outward",
                      "outward_rounded": True, "tau": V["tau"],
                      "uniform_slack": V["uniform_slack"]})
        if V["certified"] or it == max_iter:
            return theta, trace, V, bool(V["certified"])
        eps = max(BASE_MARGIN, gamma * V["L"] * V["tau"])
        idx = onp.argsort(-V["g"])[:K_WORST]
        Xc = V["X"][idx]
        pert = onp.clip(onp.repeat(Xc, PERTURBATIONS, axis=0)
                        + PERTURBATION_STD * npr.randn(PERTURBATIONS * len(Xc), 2),
                        -BOX_P, BOX_P)
        D = onp.vstack([D, Xc, pert])
        t_train = time.perf_counter()
        theta = _adam(theta, D, eps, steps=TRAINING_STEPS, lr=LEARNING_RATE)
        train_s = time.perf_counter() - t_train
    return theta, trace, V, bool(V["certified"])


def protocol_config():
    """The settings actually used by the production multi-seed experiment."""
    return {
        "max_rounds": MAX_ROUNDS, "growth_factor": GROWTH_FACTOR,
        "training_steps": TRAINING_STEPS, "learning_rate": LEARNING_RATE,
        "initial_samples": INITIAL_SAMPLES, "base_margin": BASE_MARGIN, "gamma": GAMMA,
        "worst_counterexamples": K_WORST,
        "perturbations_per_counterexample": PERTURBATIONS,
        "perturbation_std": PERTURBATION_STD,
        "optimizer": "Adam", "beta1": 0.9, "beta2": 0.999,
        "adam_epsilon": 1e-8, "optimizer_state": "reset each retraining round",
        "outcome_criterion": {
            "certified": "outward-rounded grid oracle returns certified",
            "diverged": "uncertified and max_k L_k > growth_factor * L_0",
            "stalled": "uncertified and not diverged within max_rounds",
        },
    }


def experiment(n_seeds=20, n_grid=401, depth=3, validation=None):
    """Multi-seed certified study of the non-polynomial benchmark; writes
    results/pendulum.json and paper/fig_pendulum.pdf."""
    import os, json, time
    import bounds
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HERE = os.path.dirname(os.path.abspath(__file__))
    RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
    PAPER = os.path.join(HERE, "paper"); os.makedirs(PAPER, exist_ok=True)

    print(f"=== pendulum (rate {RATE_P}, box {BOX_P}), {n_seeds} seeds, grid {n_grid} ===")
    cert = 0; div = 0; iters = []; thetas = {}
    runs = {}; timings = {}; verified = {}
    t0 = time.time()
    for seed in range(n_seeds):
        timing = []
        theta, tr, V, ok = run_pendulum(seed=seed, n_grid=n_grid, depth=depth, timings=timing)
        Lmax = max(t["L"] for t in tr)
        diverged = (not ok) and (Lmax > GROWTH_FACTOR * tr[0]["L"])
        cert += ok; div += diverged
        timings[str(seed)] = timing
        runs[str(seed)] = {"certified": bool(ok), "diverged": bool(diverged),
                           "trace": tr, "theta": theta.tolist()}
        if ok:
            iters.append(len(tr) - 1); thetas[seed] = theta; verified[seed] = V
        print(f"  seed {seed:2d}: cert={ok} iters={len(tr)-1} L={V['L']:.2f} worst={V['worst']:+.4f}")
    print(f"  runtime {time.time()-t0:.1f}s")

    # certified example: constants for a representative certified seed
    ex_seed = sorted(thetas)[0]
    th = thetas[ex_seed]
    V = verified[ex_seed]  # Same stored parameters, already checked on the entire grid.
    slack = -V["worst"]
    b = float(bounds.b_upper_bound(th, depth=depth, B=BOX_P))
    med_iters = float(onp.median(iters)) if iters else None

    out = {
        "system": "pendulum f=(x2,-sin x1 - x2)", "box": BOX_P, "rate": RATE_P,
        "n_seeds": n_seeds, "n_grid": n_grid, "depth": depth,
        "config": protocol_config(),
        "certified": cert, "diverged": div, "median_iters_to_cert": med_iters,
        "scope": "This experiment tests rate 0.2 only; no infeasibility claim at other rates.",
        "validation": validation or {},
        "verification_backend": rounding.backend_info(),
        "runs": runs,
        "example": {"seed": ex_seed, "L_cert": V["L"], "slack": slack,
                    "slack_kind": "lower bound on min negative test over stored grid",
                    "b": b, "a": float(ALPHA), "tau": V["tau"],
                    "certified": bool(V["certified"]), "margin": V["margin"],
                    "uniform_slack": V["uniform_slack"],
                    "grid_enclosed": V["grid_enclosed"], "rounding_mode": "outward",
                    "outward_rounded": V["outward_rounded"],
                    "gridpoints_covered": V["gridpoints_covered"],
                    },
    }
    with open(os.path.join(RESULTS, "pendulum.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    with open(os.path.join(RESULTS, "timing_pendulum.json"), "w") as f:
        json.dump({"units": "seconds", "note": "train_s precedes verification; bounds_s is included in verify_including_bounds_s", "runs": timings}, f, indent=2)
    print(f"  certified {cert}/{n_seeds}, diverged {div}, median iters {med_iters}")
    print(f"  example seed {ex_seed}: L={V['L']:.2f} sampled slack={slack:.4f} b={b:.3f}")

    # figure: lambda_max(C) heatmap, identity vs certified (uniformly negative)
    n = 241
    X, s = grid_p(n)
    th0 = onp.zeros((3, 3)); th0[0, 0] = 1.0; th0[2, 0] = 1.0
    g0 = onp.asarray(lam_max_C_p(th0, X)).reshape(n, n)
    g1 = onp.asarray(lam_max_C_p(th, X)).reshape(n, n)
    vmax = float(onp.max(onp.abs(g0)))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for ax, G, title in [(axes[0], g0, "identity metric (initial)"),
                         (axes[1], g1, "certified metric (CEGIS-C output)")]:
        im = ax.imshow(G, origin="lower", extent=[-BOX_P, BOX_P, -BOX_P, BOX_P],
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.contour(s, s, G, levels=[0.0], colors="k", linewidths=1.0)
        ax.set_title(title, fontsize=10); ax.set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    cb = fig.colorbar(im, ax=axes, shrink=0.9)
    cb.set_label(r"$\lambda_{\max}(C_\theta(x))$")
    plt.savefig(os.path.join(PAPER, "fig_pendulum.pdf"))
    plt.savefig(os.path.join(PAPER, "fig_pendulum.png"), dpi=160)
    plt.close()
    print(f"  wrote results/pendulum.json and paper/fig_pendulum.pdf")
    return out


if __name__ == "__main__":
    import sys
    ok1 = test_assembly()
    ok2 = test_enclosure_sound()
    print(f"\nACCEPTANCE: assembly {'PASS' if ok1 else 'FAIL'}, "
          f"enclosure {'PASS' if ok2 else 'FAIL'}")
    if not (ok1 and ok2):
        raise SystemExit(1)
    if "--experiment" in sys.argv:
        experiment(validation={"assembly": bool(ok1), "enclosure": bool(ok2)})
