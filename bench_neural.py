"""bench_neural.py (E2): a neural metric parametrization with a certified
interval-arithmetic oracle.

The Cholesky factor entries are a one-hidden-layer tanh network of the state,
    z = W1 x + b1,  h = tanh(z),  (L11, L21, L22) = W2 h + b2,
    M_theta(x) = alpha I + L(x)^T L(x).
The dynamics are the polynomial benchmark f(x) = (-x1, -(x2 - x1^2)). This study
also changes initialization, optimization settings, and the verification grid;
it does not isolate metric expressiveness or verifier conservatism.

The metric is no longer polynomial, so the coefficient-bound oracle does not
apply. We certify with hand-rolled interval arithmetic: on each subbox we enclose
x1, x2 by their ranges and propagate intervals through the exact computation of C
and its partials (tanh enclosed monotonically, its first and second derivatives
1 - h^2 and -2 h (1 - h^2) enclosed likewise), then bound the Frobenius norm of
dC/dx_k. Interval arithmetic over-approximates, so the bound is sound; subdivision
tightens it. Every verification operation is outward rounded, including weight
products, transcendental enclosures, spectral tests, and grid coverage margins.
The ordinary floating-point learner is a heuristic proposer only.
"""
import math
import time
import autograd.numpy as np
import numpy as onp

import rounding
from rounding import (Interval as Iv, add_up, add_down, mul_up, mul_down,
                      div_up, div_down, sqrt_up, sub_down)

import cegis_c
from cegis_c import ALPHA, RATE, BOX, jac, f as f_dyn

H_UNITS = 6     # hidden units
VERIFY_BLOCK_POINTS = 2_000_000
ROUND_BLOCK_POINTS = 131_072
BOX_MAX_EVALUATIONS = 2_000_000
MAX_ROUNDS = 8
GROWTH_FACTOR = 100.0
TRAINING_STEPS = 600
LEARNING_RATE = 1e-2
INITIAL_SAMPLES = 800
BASE_MARGIN = 0.05
COUPLING_FACTOR = 1.3
PERTURBATION_COUNT = 5
PERTURBATION_STD = 0.05
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
METRIC_OUTPUT_FIELDS = ("a", "b", "metric_bounds_outward_rounded",
                        "metric_bounds_depth", "raw_margin_lower",
                        "rate_lower", "overshoot_upper")


# ----------------------- numeric neural metric (autograd) -----------------------

def init_params(seed=0, scale=0.5):
    rng = onp.random.default_rng(seed)
    W1 = scale * rng.standard_normal((H_UNITS, 2))
    b1 = scale * rng.standard_normal(H_UNITS)
    W2 = scale * rng.standard_normal((3, H_UNITS))
    b2 = onp.array([1.0, 0.0, 1.0])          # bias toward identity-like L
    return _pack(W1, b1, W2, b2)


def _pack(W1, b1, W2, b2):
    return np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])


def _unpack(p):
    H = H_UNITS
    W1 = p[:2 * H].reshape(H, 2)
    b1 = p[2 * H:3 * H]
    W2 = p[3 * H:3 * H + 3 * H].reshape(3, H)
    b2 = p[3 * H + 3 * H:]
    return W1, b1, W2, b2


def L_entries(p, X):
    """L11, L21, L22 and their first spatial derivatives, from the network."""
    W1, b1, W2, b2 = _unpack(p)
    z = X @ W1.T + b1                     # (N,H)
    h = np.tanh(z)
    hp = 1.0 - h ** 2                     # tanh'
    Lv = h @ W2.T + b2                    # (N,3)
    L11, L21, L22 = Lv[:, 0], Lv[:, 1], Lv[:, 2]
    dL = []
    for k in range(2):
        dh = hp * W1[:, k]               # (N,H) d h / d x_k
        dLv = dh @ W2.T                  # (N,3)
        dL.append((dLv[:, 0], dLv[:, 1], dLv[:, 2]))
    return (L11, L21, L22), dL


def lam_max_C_n(p, X, normalized=False):
    """lambda_max of C for the neural metric on the polynomial benchmark."""
    (L11, L21, L22), dL = L_entries(p, X)
    P = ALPHA + L11 ** 2 + L21 ** 2
    Q = L21 * L22
    R = ALPHA + L22 ** 2
    # dM/dx_k
    dM = []
    for k in range(2):
        dl11, dl21, dl22 = dL[k]
        dp = 2 * L11 * dl11 + 2 * L21 * dl21
        dq = dl21 * L22 + L21 * dl22
        dr = 2 * L22 * dl22
        dM.append((dp, dq, dr))
    F = f_dyn(X)
    mdp = dM[0][0] * F[:, 0] + dM[1][0] * F[:, 1]
    mdq = dM[0][1] * F[:, 0] + dM[1][1] * F[:, 1]
    mdr = dM[0][2] * F[:, 0] + dM[1][2] * F[:, 1]
    A = jac(X)
    a11 = A[:, 0, 0]; a12 = A[:, 0, 1]; a21 = A[:, 1, 0]; a22 = A[:, 1, 1]
    s11 = 2 * (a11 * P + a21 * Q)
    s22 = 2 * (a12 * Q + a22 * R)
    s12 = a11 * Q + a21 * R + a12 * P + a22 * Q
    c11 = mdp + s11 + 2 * RATE * P
    c22 = mdr + s22 + 2 * RATE * R
    c12 = mdq + s12 + 2 * RATE * Q
    m = 0.5 * (c11 + c22)
    d = 0.5 * (c11 - c22)
    lam = m + np.sqrt(d ** 2 + c12 ** 2 + 1e-300)
    return lam / (P + R) if normalized else lam


def grid_n(n):
    s = np.linspace(-BOX, BOX, n)
    XX, YY = np.meshgrid(s, s)
    return np.stack([XX.ravel(), YY.ravel()], axis=1), s


# ----------------------- numeric derivative assembly (firewalled) -----------------------

def _network_all(p, X):
    """L entries and their first (dL) and second (d2L) spatial derivatives.
    d2L[(k,l)] = d^2 L / dx_k dx_l."""
    W1, b1, W2, b2 = _unpack(p)
    z = X @ W1.T + b1
    h = np.tanh(z)
    hp = 1.0 - h ** 2                 # tanh'
    hpp = -2.0 * h * hp               # tanh''
    Lv = h @ W2.T + b2
    L = (Lv[:, 0], Lv[:, 1], Lv[:, 2])
    dL = []
    for k in range(2):
        dLv = (hp * W1[:, k]) @ W2.T
        dL.append((dLv[:, 0], dLv[:, 1], dLv[:, 2]))
    d2L = {}
    for k in range(2):
        for l in range(2):
            d2Lv = (hpp * W1[:, k] * W1[:, l]) @ W2.T
            d2L[(k, l)] = (d2Lv[:, 0], d2Lv[:, 1], d2Lv[:, 2])
    return L, dL, d2L


def _M_derivs(L, dL, d2L):
    """M entries (P,Q,R), dM/dx_k, d2M/dx_k dx_l from L and its derivatives."""
    L11, L21, L22 = L
    P = ALPHA + L11**2 + L21**2
    Q = L21 * L22
    R = ALPHA + L22**2
    dM = []
    for k in range(2):
        a, b, c = dL[k]
        dM.append((2*L11*a + 2*L21*b, b*L22 + L21*c, 2*L22*c))
    d2M = {}
    for k in range(2):
        for l in range(2):
            a_k, b_k, c_k = dL[k]; a_l, b_l, c_l = dL[l]; a_kl, b_kl, c_kl = d2L[(k, l)]
            dP = 2*(a_k*a_l + L11*a_kl) + 2*(b_k*b_l + L21*b_kl)
            dQ = b_kl*L22 + b_k*c_l + b_l*c_k + L21*c_kl
            dR = 2*(c_k*c_l + L22*c_kl)
            d2M[(k, l)] = (dP, dQ, dR)
    return (P, Q, R), dM, d2M


def C_and_dC(p, X):
    """C entries (c11,c12,c22) and dC/dx_k for k=0,1, numeric arrays."""
    L, dL, d2L = _network_all(p, X)
    (P, Q, R), dM, d2M = _M_derivs(L, dL, d2L)
    F = f_dyn(X); A = jac(X)
    a11 = A[:, 0, 0]; a12 = A[:, 0, 1]; a21 = A[:, 1, 0]; a22 = A[:, 1, 1]
    # dA/dx_k: only dA21/dx1 = 2
    dA21 = [2.0, 0.0]
    # df_l/dx_k = A_{l,k}
    def Cmat(P, Q, R, dM):
        mdp = dM[0][0]*F[:, 0] + dM[1][0]*F[:, 1]
        mdq = dM[0][1]*F[:, 0] + dM[1][1]*F[:, 1]
        mdr = dM[0][2]*F[:, 0] + dM[1][2]*F[:, 1]
        s11 = 2*(a11*P + a21*Q); s22 = 2*(a12*Q + a22*R)
        s12 = a11*Q + a21*R + a12*P + a22*Q
        return (mdp + s11 + 2*RATE*P, mdq + s12 + 2*RATE*Q, mdr + s22 + 2*RATE*R)
    c11, c12, c22 = Cmat(P, Q, R, dM)
    dC = []
    for k in range(2):
        dP, dQ, dR = dM[k]
        # d(Mdot)/dx_k = sum_l [ d2M_{k,l} f_l + dM_l * A_{l,k} ]
        dmd = [0.0, 0.0, 0.0]
        for l in range(2):
            for e in range(3):
                dmd[e] = dmd[e] + d2M[(k, l)][e]*F[:, l] + dM[l][e]*A[:, l, k]
        # d(A^T M + M A)/dx_k = (dA)^T M + A^T dM + dM A + M dA
        # dA has only entry (2,1)=dA21[k]
        dA11, dA12, dA22 = 0.0, 0.0, 0.0
        da21 = dA21[k]
        ds11 = 2*(a11*dP + a21*dQ) + 2*(da21*Q)
        ds22 = 2*(a12*dQ + a22*dR)
        ds12 = a11*dQ + a21*dR + a12*dP + a22*dQ + da21*R
        dc11 = dmd[0] + ds11 + 2*RATE*dP
        dc12 = dmd[1] + ds12 + 2*RATE*dQ
        dc22 = dmd[2] + ds22 + 2*RATE*dR
        dC.append((dc11, dc12, dc22))
    return (c11, c12, c22), dC, (P, R)


def _test_derivs(seed=0, n=6, tol=1e-5):
    """Firewall: analytic dC/dx_k matches central finite differences of C."""
    rng = onp.random.default_rng(seed)
    maxerr = 0.0
    for _ in range(n):
        p = init_params(rng.integers(1000))
        X = rng.uniform(-BOX + 0.2, BOX - 0.2, size=(50, 2))
        (c11, c12, c22), dC, _ = C_and_dC(p, X)
        hd = 1e-6
        for k in range(2):
            e = onp.zeros(2); e[k] = hd
            (c11p, c12p, c22p), _, _ = C_and_dC(p, X + e)
            (c11m, c12m, c22m), _, _ = C_and_dC(p, X - e)
            for num, cp, cm in [(dC[k][0], c11p, c11m), (dC[k][1], c12p, c12m), (dC[k][2], c22p, c22m)]:
                fd = (cp - cm) / (2 * hd)
                maxerr = max(maxerr, float(onp.max(onp.abs(num - fd))))
    print(f"[neural dC firewall] max |analytic - FD| = {maxerr:.2e}")
    return maxerr < tol


# ----------------------- interval arithmetic (certified oracle) -----------------------

def _iv_network_all(p, x1, x2, second=True):
    W1, b1, W2, b2 = _unpack(p)
    h = []; hp = []; hpp = []
    for n in range(H_UNITS):
        z = x1*float(W1[n, 0]) + x2*float(W1[n, 1]) + float(b1[n])
        hn = z.tanh(); h.append(hn)
        hpn = Iv(1.0) - hn.square()
        hpn = Iv(onp.maximum(0.0, hpn.lo), onp.minimum(1.0, hpn.hi))
        hp.append(hpn)
        if second:
            hpp.append(hn*hpn*(-2.0))

    def out(coeffs, arr):  # sum_n coeffs[n]*arr[n] + bias
        acc = None
        for n in range(H_UNITS):
            term = arr[n]*coeffs[n]
            acc = term if acc is None else acc + term
        return acc
    L = tuple(out(W2[r], h) + float(b2[r]) for r in range(3))
    dL = []
    for k in range(2):
        dL.append(tuple(out([Iv(float(W2[r, n])) * float(W1[n, k])
                             for n in range(H_UNITS)], hp) for r in range(3)))
    d2L = {}
    if second:
        for k in range(2):
            for l in range(2):
                d2L[(k, l)] = tuple(out(
                    [Iv(float(W2[r, n])) * float(W1[n, k]) * float(W1[n, l])
                     for n in range(H_UNITS)], hpp) for r in range(3))
    return L, dL, d2L


def _interval_C_and_dC(p, x1, x2):
    """Outward enclosures from the original binary64 weights, before products."""
    L, dL, d2L = _iv_network_all(p, x1, x2)
    L11, L21, L22 = L
    P = L11.square() + L21.square() + ALPHA
    Q = L21*L22
    R = L22.square() + ALPHA
    dM = []
    for k in range(2):
        a, b, cc = dL[k]
        dM.append((L11*a*2.0 + L21*b*2.0, b*L22 + L21*cc, L22*cc*2.0))
    d2M = {}
    for k in range(2):
        for l in range(2):
            a_k, b_k, c_k = dL[k]; a_l, b_l, c_l = dL[l]; a_kl, b_kl, c_kl = d2L[(k, l)]
            d2M[(k, l)] = (a_k*a_l*2.0 + L11*a_kl*2.0 + b_k*b_l*2.0 + L21*b_kl*2.0,
                           b_kl*L22 + b_k*c_l + b_l*c_k + L21*c_kl,
                           c_k*c_l*2.0 + L22*c_kl*2.0)
    # dynamics as intervals: f=(-x1, -(x2-x1^2)); A=[[-1,0],[2x1,-1]]
    F = [(-x1), -(x2 - x1.square())]
    Aiv = [[Iv(-onp.ones_like(x1.lo), -onp.ones_like(x1.lo)), Iv(onp.zeros_like(x1.lo), onp.zeros_like(x1.lo))],
           [x1*2.0, Iv(-onp.ones_like(x1.lo), -onp.ones_like(x1.lo))]]

    def Cmat(P, Q, R, dM):
        mdp = dM[0][0]*F[0] + dM[1][0]*F[1]
        mdq = dM[0][1]*F[0] + dM[1][1]*F[1]
        mdr = dM[0][2]*F[0] + dM[1][2]*F[1]
        s11 = (Aiv[0][0]*P + Aiv[1][0]*Q)*2.0
        s22 = (Aiv[0][1]*Q + Aiv[1][1]*R)*2.0
        s12 = Aiv[0][0]*Q + Aiv[1][0]*R + Aiv[0][1]*P + Aiv[1][1]*Q
        return (mdp + s11 + P*Iv(RATE)*2.0, mdq + s12 + Q*Iv(RATE)*2.0, mdr + s22 + R*Iv(RATE)*2.0)
    c11, c12, c22 = Cmat(P, Q, R, dM)
    dA21 = [2.0, 0.0]
    dCabs = []
    for k in range(2):
        dP, dQ, dR = dM[k]
        dmd = [None, None, None]
        for l in range(2):
            for e in range(3):
                term = d2M[(k, l)][e]*F[l] + dM[l][e]*Aiv[l][k]
                dmd[e] = term if dmd[e] is None else dmd[e] + term
        da21 = dA21[k]
        ds11 = (Aiv[0][0]*dP + Aiv[1][0]*dQ)*2.0 + Q*(2*da21)
        ds22 = (Aiv[0][1]*dQ + Aiv[1][1]*dR)*2.0
        ds12 = Aiv[0][0]*dQ + Aiv[1][0]*dR + Aiv[0][1]*dP + Aiv[1][1]*dQ + R*da21
        dc11 = dmd[0] + ds11 + dP*Iv(RATE)*2.0
        dc12 = dmd[1] + ds12 + dQ*Iv(RATE)*2.0
        dc22 = dmd[2] + ds22 + dR*Iv(RATE)*2.0
        dCabs.append((dc11, dc12, dc22))
    return (c11, c12, c22), dCabs, (P, Q, R), dM


def _certified_C(p, depth):
    """Outward C, spatial derivatives, and trace bounds on dyadic subboxes."""
    nb = 2 ** depth
    hh = BOX / nb
    cs = onp.array([-BOX + (2*i + 1)*hh for i in range(nb)])
    c1, c2 = [a.ravel() for a in onp.meshgrid(cs, cs)]
    # BOX and dyadic centers/halfwidths are exactly representable here.
    x1 = Iv(c1 - hh, c1 + hh); x2 = Iv(c2 - hh, c2 + hh)
    C, dC, M, dM = _interval_C_and_dC(p, x1, x2)
    dCabs = [tuple(a.absmax() for a in row) for row in dC]
    cF2 = (C[0].square() + C[1].square()*2.0 + C[2].square()).hi
    v_lo = (M[0] + M[2]).lo
    dvabs = tuple((Iv(row[0].absmax()) + row[2].absmax()).hi for row in dM)
    return dCabs, cF2, v_lo, M, dvabs


def L_raw_n(p, depth=3):
    dCabs, _, _, _, _ = _certified_C(p, depth)
    total = Iv(0.0)
    for d11, d12, d22 in dCabs:
        total = total + float(onp.max((Iv(d11).square() + Iv(d12).square()*2.0 + Iv(d22).square()).hi))
    return float(total.sqrt().hi)


def L_norm_n(p, depth=3):
    dCabs, cF2, v_lo, _, dvabs = _certified_C(p, depth)
    floor = float((Iv(ALPHA)*2.0).lo)
    v_min = max(floor, float(onp.min(v_lo)))
    total = Iv(0.0)
    for d11, d12, d22 in dCabs:
        total = total + float(onp.max((Iv(d11).square() + Iv(d12).square()*2.0 + Iv(d22).square()).hi))
    Lg = float(total.sqrt().hi)
    Lv = float((Iv(float(onp.max(dvabs[0]))).square() +
                Iv(float(onp.max(dvabs[1]))).square()).sqrt().hi)
    gt_sup = float(onp.max((Iv(cF2).sqrt() / Iv(onp.maximum(floor, v_lo))).hi))
    return div_up(add_up(Lg, mul_up(gt_sup, Lv)), v_min)


def metric_bounds_n(p, depth=3):
    """Return rigorous global a,b with a I <= M(x) <= b I on the full box.

    The Cholesky floor gives a exactly. Each dyadic subbox contributes an
    outward Frobenius bound on its interval metric; the maximum covers the
    whole domain, including subbox boundaries. No grid samples enter b.
    """
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
        raise ValueError("metric-bound subdivision depth must be a nonnegative integer")
    _, _, _, M, _ = _certified_C(p, depth)
    norm = (M[0].square() + M[1].square()*2.0 + M[2].square()).sqrt()
    b = float(onp.max(norm.hi))
    a = float(ALPHA)
    if not (math.isfinite(a) and math.isfinite(b) and 0 < a <= b):
        raise ArithmeticError("invalid certified neural metric bounds")
    return {"a": a, "b": b, "metric_bounds_outward_rounded": True,
            "metric_bounds_depth": depth}


def certified_metric_outputs_n(p, uniform_slack, normalized=True, depth=3):
    """Quantitative outputs conditional on a certified nonnegative slack.

    Rates and raw margins round down; the Euclidean differential overshoot
    rounds up. Finite-trajectory use needs the paper's connecting-flow
    hypothesis in addition to these metric and matrix inequalities.
    """
    slack = float(uniform_slack)
    if not math.isfinite(slack) or slack < 0:
        raise ValueError("quantitative neural outputs require certified nonnegative slack")
    out = metric_bounds_n(p, depth)
    eta = max(0.0, mul_down(mul_down(2.0, out["a"]), slack)) if normalized else slack
    improved_rate = max(float(RATE), add_down(RATE, div_down(eta, mul_up(2.0, out["b"]))))
    out.update(raw_margin_lower=eta, rate_lower=improved_rate,
               overshoot_upper=sqrt_up(div_up(out["b"], out["a"])))
    return out


def interval_test_n(p, x1, x2, normalized=True):
    """Outward contraction test on a point or box of exact binary64 endpoints."""
    L, dL, _ = _iv_network_all(p, x1, x2, second=False)
    l11, l21, l22 = L
    P = l11.square() + l21.square() + ALPHA
    Q = l21*l22
    R = l22.square() + ALPHA
    dM = []
    for a, b, c in dL:
        dM.append((l11*a*2.0 + l21*b*2.0,
                   b*l22 + l21*c, l22*c*2.0))
    f1, f2 = -x1, -(x2 - x1.square())
    mdp = dM[0][0]*f1 + dM[1][0]*f2
    mdq = dM[0][1]*f1 + dM[1][1]*f2
    mdr = dM[0][2]*f1 + dM[1][2]*f2
    c11 = mdp + (P*(-1.0) + x1*2.0*Q)*2.0 + P*Iv(RATE)*2.0
    c22 = mdr + R*(-2.0) + R*Iv(RATE)*2.0
    c12 = mdq + Q*(-1.0) + x1*2.0*R + Q*(-1.0) + Q*Iv(RATE)*2.0
    result = rounding.lambda_max_2x2(c11, c12, c22)
    return result/(P + R) if normalized else result


def grid_enclosure_n(p, X, normalized=True):
    """Outward contraction test at stored grid coordinates; no libm assumption."""
    return interval_test_n(p, Iv(X[:, 0]), Iv(X[:, 1]), normalized=normalized)


def _top_violations(points, values, count):
    """Largest values, breaking exact ties by their grid traversal order."""
    if len(values) > count:
        cutoff = onp.partition(values, len(values) - count)[len(values) - count]
        above = onp.flatnonzero(values > cutoff)
        tied = onp.flatnonzero(values == cutoff)[:count - len(above)]
        selected = onp.concatenate([above, tied])
    else:
        selected = onp.arange(len(values))
    selected = selected[onp.argsort(-values[selected], kind="stable")]
    return points[selected], values[selected]


def _rounded_grid_boxes(p, nodes, margin, normalized=True, max_evaluations=BOX_MAX_EVALUATIONS, target=None):
    """Prove a stored Cartesian grid by disjoint inclusive index rectangles.

    Each accepted interval rectangle proves every grid point it contains at
    the ORIGINAL outward Lipschitz threshold. Unresolved rectangles split into
    disjoint index halves until individual points. A cap requests the exact
    streaming point fallback; it never permits a nominal certificate.
    """
    target=margin if target is None else min(margin,float(target))
    if not math.isfinite(target):raise ValueError("nonfinite grid grouping target")
    count=len(nodes)
    if count<1 or count>math.isqrt(onp.iinfo(onp.int64).max):
        raise ValueError("grid index cardinality exceeds exact int64 proof accounting")
    pending=onp.array([[0,count-1,0,count-1]],dtype=onp.int64)
    covered=0; evaluated=0; worst=-onp.inf
    while len(pending):
        if evaluated+len(pending)>max_evaluations:
            return {"complete":False,"fallback":True,"boxes_evaluated":evaluated,
                    "gridpoints_covered":covered}
        x0,x1,y0,y1=pending.T
        upper=onp.asarray(interval_test_n(
            p,Iv(nodes[x0],nodes[x1]),Iv(nodes[y0],nodes[y1]),normalized=normalized).hi)
        if not onp.isfinite(upper).all():
            raise ValueError("nonfinite outward neural box bound")
        evaluated+=len(pending)
        # A strict threshold avoids an inconclusive downward-rounded zero slack.
        singleton=(x0==x1)&(y0==y1)
        # A heuristic target may request tighter boxes; individual grid nodes
        # always use the original certified threshold, never the effort guide.
        accepted=(upper<target)|(singleton&(upper<margin))
        sizes=(x1-x0+1)*(y1-y0+1)
        covered+=int(sizes[accepted].sum())
        if accepted.any():worst=max(worst,float(upper[accepted].max()))
        failed=singleton&(~accepted)
        if failed.any():
            # A negative verdict needs no exhaustive proof. These interval
            # failures are candidate counterexamples, not claims of instability.
            bad=onp.flatnonzero(failed)
            j=bad[onp.argmax(upper[bad])]
            return {"complete":False,"fallback":False,"certified":False,
                    "worst":max(worst,float(upper[j])),
                    "viol":onp.array([[nodes[x0[j]],nodes[y0[j]]]]),
                    "boxes_evaluated":evaluated,"gridpoints_covered":covered+int(failed.sum())}
        unresolved=pending[~accepted]
        if not len(unresolved):break
        split_x=(unresolved[:,1]-unresolved[:,0]) >= (unresolved[:,3]-unresolved[:,2])
        left=unresolved.copy();right=unresolved.copy()
        mid=(unresolved[:,0]+unresolved[:,1])//2
        left[split_x,1]=mid[split_x];right[split_x,0]=mid[split_x]+1
        mid=(unresolved[:,2]+unresolved[:,3])//2
        left[~split_x,3]=mid[~split_x];right[~split_x,2]=mid[~split_x]+1
        if onp.any(left[:,0]>left[:,1]) or onp.any(left[:,2]>left[:,3]) or onp.any(right[:,0]>right[:,1]) or onp.any(right[:,2]>right[:,3]):
            raise ArithmeticError("invalid disjoint grid partition")
        # Children cover their parent's stored indices without sharing a node.
        parent_size=(unresolved[:,1]-unresolved[:,0]+1)*(unresolved[:,3]-unresolved[:,2]+1)
        child_size=(left[:,1]-left[:,0]+1)*(left[:,3]-left[:,2]+1)+(right[:,1]-right[:,0]+1)*(right[:,3]-right[:,2]+1)
        if not onp.array_equal(parent_size,child_size):
            raise ArithmeticError("grid partition lost or duplicated indices")
        pending=onp.concatenate([left,right])
    if covered!=count**2:
        raise ArithmeticError("outward grid proof has incomplete index coverage")
    return {"complete":True,"fallback":False,"certified":True,"worst":worst,
            "viol":onp.zeros((0,2)),"boxes_evaluated":evaluated,"gridpoints_covered":covered}


def verify_n(p, n_grid, normalized=True, depth=3, k_worst=60):
    """Heuristic rejection screen followed by an outward proof of the same grid."""
    if n_grid<2 or k_worst<1:
        raise ValueError("n_grid must be >= 2 and k_worst must be positive")
    s=onp.linspace(-BOX,BOX,n_grid)
    if s[0]!=-BOX or s[-1]!=BOX:
        raise ValueError("grid does not include the exact domain endpoints")
    tau=rounding.grid_radius(s,dimension=2)
    t_bounds=time.perf_counter()
    L=L_norm_n(p,depth) if normalized else L_raw_n(p,depth)
    bounds_s=time.perf_counter()-t_bounds
    if not math.isfinite(L) or L<0:raise ValueError("invalid Lipschitz bound")
    margin=-mul_up(L,tau)
    worst=-onp.inf
    cand_pts=onp.zeros((0,2));cand_g=onp.zeros(0)
    block=max(1,VERIFY_BLOCK_POINTS//n_grid)
    for r0 in range(0,n_grid,block):
        rows=s[r0:r0+block]
        pts=onp.stack([onp.tile(s,len(rows)),onp.repeat(rows,n_grid)],axis=1)
        g=onp.asarray(lam_max_C_n(p,pts,normalized=normalized))
        if not onp.isfinite(g).all():raise ValueError("nonfinite neural test value")
        worst=max(worst,float(g.max()))
        mask=g>margin
        if mask.any():
            lp,lg=_top_violations(pts[mask],g[mask],k_worst)
            cand_pts,cand_g=_top_violations(onp.concatenate([cand_pts,lp]),onp.concatenate([cand_g,lg]),k_worst)
    nominal_worst=worst
    attempted=len(cand_pts)==0
    grid_enclosed=False;box_evaluations=0;point_evaluations=0;covered=0
    method="nominal_rejection_only"
    target=margin
    if attempted:
        # The nominal value chooses effort only. Every accepted enclosure is
        # checked against the original margin, and singleton fallback ignores
        # this stronger grouping target. It is not a trusted numerical bound.
        target=min(margin,nominal_worst*0.5+margin*0.5)
        proof=_rounded_grid_boxes(p,s,margin,normalized=normalized,target=target)
        box_evaluations=proof["boxes_evaluated"]
        covered=proof["gridpoints_covered"]
        if proof["fallback"]:
            method="streaming_point_intervals"
            worst=-onp.inf
            block=max(1,ROUND_BLOCK_POINTS//n_grid)
            for r0 in range(0,n_grid,block):
                rows=s[r0:r0+block]
                pts=onp.stack([onp.tile(s,len(rows)),onp.repeat(rows,n_grid)],axis=1)
                upper=onp.asarray(grid_enclosure_n(p,pts,normalized=normalized).hi)
                if not onp.isfinite(upper).all():raise ValueError("nonfinite outward neural grid bound")
                point_evaluations+=len(pts)
                worst=max(worst,float(upper.max()))
                mask=upper>margin
                if mask.any():
                    lp,lg=_top_violations(pts[mask],upper[mask],k_worst)
                    cand_pts,cand_g=_top_violations(onp.concatenate([cand_pts,lp]),onp.concatenate([cand_g,lg]),k_worst)
            covered=n_grid**2;grid_enclosed=True
        else:
            method="disjoint_index_rectangles"
            worst=proof["worst"];cand_pts=proof["viol"]
            grid_enclosed=proof["complete"]
    uniform_slack=sub_down(-worst,-margin)
    certified=grid_enclosed and len(cand_pts)==0 and uniform_slack>=0.0
    if not certified and worst<=margin:
        worst=float(onp.nextafter(margin,onp.inf))
        uniform_slack=sub_down(-worst,-margin)
        if not len(cand_pts):cand_pts=onp.array([[s[0],s[0]]])
    result = {"certified":bool(certified),"viol":cand_pts,"margin":margin,"L":L,"tau":tau,
            "worst":worst,"nominal_worst":nominal_worst,"bounds_s":bounds_s,
            "grid_enclosed":bool(grid_enclosed),"strict_grid_attempted":bool(attempted),
            "rounding_mode":"outward","outward_rounded":True,"grid_check_method":method,
            "worst_kind":"outward_grid_upper" if grid_enclosed else "diagnostic_rejection",
            "uniform_slack":uniform_slack,"nominal_grid_evaluations":n_grid**2,
            "outward_grid_evaluations":point_evaluations,"outward_box_evaluations":box_evaluations,
            "gridpoints_covered":covered,"grouping_target":target,
            "oracle_evaluations":n_grid**2+point_evaluations+box_evaluations}
    if certified:
        t_metric = time.perf_counter()
        result.update(certified_metric_outputs_n(p, uniform_slack, normalized, depth))
        result["bounds_s"] += time.perf_counter()-t_metric
    return result


def test_enclosure(seed=1, n_p=6, n_grid=401, depth=3):
    """Interval bounds enclose the true sup; certified L_norm >= empirical."""
    rng = onp.random.default_rng(seed)
    fails = 0; ratios = []
    X, _ = grid_n(n_grid)
    for _ in range(n_p):
        p = init_params(rng.integers(10000))
        dCabs, cF2, v_lo, _, _ = _certified_C(p, depth)
        (c11, c12, c22), dC, _ = C_and_dC(p, X)
        # sup |dc11/dx1| certified >= dense
        cert = float(onp.max(dCabs[0][0]))
        dense = float(onp.max(onp.abs(dC[0][0])))
        if cert + 1e-9 < dense:
            fails += 1
        Lc = L_norm_n(p, depth)
        hstep = 1e-5; g0 = lam_max_C_n(p, X, normalized=True)
        gx = (lam_max_C_n(p, X + onp.array([hstep, 0.]), normalized=True) - g0)/hstep
        gy = (lam_max_C_n(p, X + onp.array([0., hstep]), normalized=True) - g0)/hstep
        Lemp = float(onp.max(onp.sqrt(gx**2 + gy**2)))
        ratios.append(Lc/Lemp if Lemp > 0 else float("inf"))
        if Lc + 1e-9 < Lemp:
            fails += 1
    print(f"[neural enclosure] fails={fails}, cert/emp L ratio "
          f"min={min(ratios):.1f} median={onp.median(ratios):.1f} max={max(ratios):.1f}")
    return fails == 0


# ----------------------- learner and experiment -----------------------

from autograd import grad as _grad
import numpy.random as npr


def _loss(p, X, eps):
    return np.mean(np.maximum(0.0, lam_max_C_n(p, X, normalized=True) + eps))


_dloss = _grad(_loss)


def _adam(p, X, eps, steps=TRAINING_STEPS, lr=LEARNING_RATE):
    m = onp.zeros_like(p); v = onp.zeros_like(p)
    b1, b2, e = ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON
    for t in range(1, steps + 1):
        g = _dloss(p, X, eps)
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g ** 2
        p = p - lr * (m / (1 - b1 ** t)) / (onp.sqrt(v / (1 - b2 ** t)) + e)
    return p


def run_neural(seed=0, n_grid=8001, max_iter=MAX_ROUNDS, gamma=COUPLING_FACTOR, depth=3, timings=None):
    """CEGIS-C with the neural metric, normalized coupled variant."""
    npr.seed(seed)
    D = npr.uniform(-BOX, BOX, size=(INITIAL_SAMPLES, 2))
    p = init_params(seed)
    eps = BASE_MARGIN
    trace = []
    train_s = 0.0
    for it in range(max_iter + 1):
        t_verify = time.perf_counter()
        V = verify_n(p, n_grid, normalized=True, depth=depth)
        verify_s = time.perf_counter() - t_verify
        if timings is not None:
            timings.append({"iter": it, "train_s": train_s,
                            "verify_including_bounds_s": verify_s,
                            "bounds_s": V["bounds_s"],
                            "oracle_evaluations": V["oracle_evaluations"],
                            "nominal_grid_evaluations": V["nominal_grid_evaluations"],
                            "outward_grid_evaluations": V["outward_grid_evaluations"],
                            "outward_box_evaluations": V["outward_box_evaluations"],
                            "gridpoints_covered": V["gridpoints_covered"]})
        trace.append({"iter": it, "worst": V["worst"], "L": V["L"],
                      "gap": V["worst"] - V["margin"],
                      "oracle_evaluations": V["oracle_evaluations"],
                      "nominal_grid_evaluations": V["nominal_grid_evaluations"],
                      "outward_grid_evaluations": V["outward_grid_evaluations"],
                      "outward_box_evaluations": V["outward_box_evaluations"],
                      "gridpoints_covered": V["gridpoints_covered"],
                      "grid_check_method": V["grid_check_method"],
                      "nominal_worst": V["nominal_worst"],
                      "grouping_target": V.get("grouping_target", V["margin"]),
                      "grid_enclosed": V["grid_enclosed"], "rounding_mode": "outward",
                      "outward_rounded": True, "tau": V["tau"],
                      "uniform_slack": V["uniform_slack"]})
        if V["certified"] or it == max_iter:
            return p, trace, V, bool(V["certified"])
        eps = max(BASE_MARGIN, gamma * V["L"] * V["tau"])
        Xc = V["viol"]
        pert = onp.clip(onp.repeat(Xc, PERTURBATION_COUNT, axis=0) +
                        PERTURBATION_STD * npr.randn(PERTURBATION_COUNT * len(Xc), 2),
                        -BOX, BOX)
        D = onp.vstack([D, Xc, pert])
        t_train = time.perf_counter()
        p = _adam(p, D, eps)
        train_s = time.perf_counter() - t_train
    return p, trace, V, bool(V["certified"])


def experiment(n_seeds=20, op_grid=8001, coarse_grid=2001, depth=3, validation=None,
               max_iter=MAX_ROUNDS, gamma=COUPLING_FACTOR):
    """Neural-metric study with its own optimizer, initialization, and grid.
    Writes results/neural.json and paper/fig_neural.pdf."""
    import os, json, time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HERE = os.path.dirname(os.path.abspath(__file__))
    RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
    PAPER = os.path.join(HERE, "paper"); os.makedirs(PAPER, exist_ok=True)

    print(f"=== neural metric ({H_UNITS} tanh units), {n_seeds} seeds ===")
    t0 = time.time()
    op_cert = 0; op_div = 0; iters = []; thetas = {}
    coarse_cert = 0
    runs = {}; timings = {}; verified = {}
    for seed in range(n_seeds):
        timing = []
        p, tr, V, ok = run_neural(seed=seed, n_grid=op_grid, depth=depth, timings=timing,
                                 max_iter=max_iter, gamma=gamma)
        Lmax = max(t["L"] for t in tr)
        diverged = (not ok) and (Lmax > GROWTH_FACTOR * tr[0]["L"])
        op_cert += ok; op_div += diverged
        if ok:
            iters.append(len(tr) - 1); thetas[seed] = p; verified[seed] = V
        Vc = verify_n(p, coarse_grid, normalized=True, depth=depth) if ok else None
        coarse_cert += 1 if (Vc and Vc["certified"]) else 0
        timings[str(seed)] = timing
        runs[str(seed)] = {"certified": bool(ok), "diverged": bool(diverged),
                           "stalled": bool(not ok and not diverged),
                           "certified_coarse": bool(Vc and Vc["certified"]),
                           "coarse_grid_enclosed": bool(Vc and Vc["grid_enclosed"]),
                           "coarse_uniform_slack": Vc["uniform_slack"] if Vc else None,
                           "coarse_gridpoints_covered": Vc["gridpoints_covered"] if Vc else 0,
                           "trace": tr, "parameters": p.tolist()}
        if ok:
            runs[str(seed)].update({key: V[key] for key in METRIC_OUTPUT_FIELDS})
        print(f"  seed {seed:2d}: op(grid {op_grid})={ok} iters={len(tr)-1} "
              f"L={V['L']:.1f} worst={V['worst']:+.3f}")
    print(f"  runtime {time.time()-t0:.0f}s")

    ex_seed = sorted(thetas)[0]
    pe = thetas[ex_seed]
    V = verified[ex_seed]  # Reuse the completed strict check, without a duplicate full-grid pass.
    slack = -V["worst"]
    out = {
        "metric": f"one-hidden-layer tanh network, {H_UNITS} units",
        "units": H_UNITS,
        "dynamics": "polynomial benchmark f=(-x1,-(x2-x1^2))",
        "n_seeds": n_seeds, "op_grid": op_grid, "coarse_grid": coarse_grid,
        "depth": depth,
        "config": {"max_rounds": max_iter, "growth_factor": GROWTH_FACTOR,
                   "target_rate": float(RATE), "metric_floor": float(ALPHA),
                   "training_steps": TRAINING_STEPS, "learning_rate": LEARNING_RATE,
                   "initial_samples": INITIAL_SAMPLES, "base_margin": BASE_MARGIN,
                   "gamma": gamma, "perturbations_per_counterexample": PERTURBATION_COUNT,
                   "perturbation_std": PERTURBATION_STD, "optimizer": "Adam",
                   "optimizer_state": "reset each retraining round",
                   "beta1": ADAM_BETA1, "beta2": ADAM_BETA2,
                   "adam_epsilon": ADAM_EPSILON,
                   "outcome_criterion": {
                       "certified": "outward-rounded grid oracle returns certified",
                       "diverged": "uncertified and max_k L_k > growth_factor * L_0",
                       "stalled": "uncertified and not diverged within max_rounds"}},
        "validation": validation or {},
        "verification_backend": rounding.backend_info(),
        "runs": runs,
        "certified_op": op_cert, "diverged_op": op_div,
        "certified_coarse": coarse_cert,
        "median_iters_to_cert": float(onp.median(iters)) if iters else None,
        "example": {"seed": ex_seed, "L_cert": V["L"], "slack": slack,
                    **{key: V[key] for key in METRIC_OUTPUT_FIELDS},
                    "slack_kind": "lower bound on min negative test over stored grid",
                    "tau": V["tau"], "certified": bool(V["certified"]),
                    "margin": V["margin"],
                    "uniform_slack": V["uniform_slack"],
                    "grid_enclosed": V["grid_enclosed"], "rounding_mode": "outward",
                    "outward_rounded": V["outward_rounded"],
                    "gridpoints_covered": V["gridpoints_covered"],
                    "outward_box_evaluations": V["outward_box_evaluations"],
                    "grid_check_method": V["grid_check_method"],
                    "nominal_worst": V["nominal_worst"],
                      "grouping_target": V.get("grouping_target", V["margin"]),
                    },
    }
    with open(os.path.join(RESULTS, "neural.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    with open(os.path.join(RESULTS, "timing_neural.json"), "w") as f:
        json.dump({"units": "seconds", "note": "train_s precedes verification; bounds_s is included in verify_including_bounds_s; coarse-grid postchecks are excluded", "runs": timings}, f, indent=2)
    print(f"  certified {op_cert}/{n_seeds} at grid {op_grid}, "
          f"{coarse_cert}/{n_seeds} of those also at grid {coarse_grid}; "
          f"diverged {op_div}; median iters {out['median_iters_to_cert']}")

    # figure: certified neural-metric condition heatmap
    n = 241
    X, s = grid_n(n)
    p0 = init_params(0)
    g0 = onp.asarray(lam_max_C_n(p0, X)).reshape(n, n)
    g1 = onp.asarray(lam_max_C_n(pe, X)).reshape(n, n)
    vmax = float(onp.max(onp.abs(g1))) * 1.2
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for ax, G, title in [(axes[0], g0, "untrained neural metric"),
                         (axes[1], g1, "certified neural metric (CEGIS-C output)")]:
        im = ax.imshow(G, origin="lower", extent=[-BOX, BOX, -BOX, BOX],
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.contour(s, s, G, levels=[0.0], colors="k", linewidths=1.0)
        ax.set_title(title, fontsize=10); ax.set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    cb = fig.colorbar(im, ax=axes, shrink=0.9)
    cb.set_label(r"$\lambda_{\max}(C_\theta(x))$")
    plt.savefig(os.path.join(PAPER, "fig_neural.pdf"))
    plt.savefig(os.path.join(PAPER, "fig_neural.png"), dpi=160)
    plt.close()
    print("  wrote results/neural.json and paper/fig_neural.pdf")
    return out


if __name__ == "__main__":
    import sys
    ok1 = _test_derivs()
    ok2 = test_enclosure()
    print(f"\nACCEPTANCE: dC firewall {'PASS' if ok1 else 'FAIL'}, "
          f"enclosure {'PASS' if ok2 else 'FAIL'}")
    if not (ok1 and ok2):
        raise SystemExit(1)
    if "--experiment" in sys.argv:
        experiment(validation={"derivatives": bool(ok1), "enclosure": bool(ok2)})
