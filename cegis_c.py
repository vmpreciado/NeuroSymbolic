"""CEGIS-C: counterexample-guided synthesis of a contraction metric.

System: f(x) = (-x1, -(x2 - x1^2)) on X = [-2,2]^2.
Ground truth: contracting with rate 1 in M*(x) = J_phi^T J_phi, phi=(x1, x2+x1^2).
Identity metric fails on |x1| > 1.

Metric class: M_theta(x) = alpha*I + L(x)^T L(x), L lower-triangular,
entries linear in features [1, x1, x2]. Theta in R^{3x3} (3 entries x 3 features).

Verifier: uniform grid with Lipschitz margin (Prop. 1 of the paper). The
oracle's Lipschitz constant is an outward-rounded upper bound from bounds.py
(exact rational coefficient assembly and subdivision). Positive answers also
require outward-rounded grid eigenvalue and quotient checks. The empirical sampled estimator remains available
under verify(..., certified=False) for comparison plots only; it is not
sound and is not used to claim any certificate.
"""
import autograd.numpy as np
from autograd import grad
import numpy.random as npr
import json, time

npr.seed(0)

ALPHA = 0.1          # metric floor: M >= ALPHA*I by construction
RATE = 0.4           # target contraction rate lambda_c
BOX = 2.0            # domain [-BOX, BOX]^2
EPS = 0.05           # learner margin epsilon
LIP_SAFETY = 1.5     # safety factor on the empirical estimator (diagnostic only;
                     # never enters a certificate, see lipschitz_estimate)

# ---------------- dynamics ----------------
def f(X):            # X: (N,2)
    return np.stack([-X[:, 0], -(X[:, 1] - X[:, 0] ** 2)], axis=1)

def jac(X):          # A(x): (N,2,2)
    N = X.shape[0]
    A = np.zeros((N, 2, 2))
    A = np.stack([
        np.stack([-np.ones(N), np.zeros(N)], axis=1),
        np.stack([2.0 * X[:, 0], -np.ones(N)], axis=1)
    ], axis=1)
    return A

# ---------------- metric class ----------------
# features phi(x) = [1, x1, x2]; dphi/dx1 = [0,1,0]; dphi/dx2 = [0,0,1]
def feats(X):
    N = X.shape[0]
    return np.stack([np.ones(N), X[:, 0], X[:, 1]], axis=1)  # (N,3)

DPHI = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]

def L_of(theta, PH):                       # theta: (3,3) rows -> L11, L21, L22
    l11 = PH @ theta[0]
    l21 = PH @ theta[1]
    l22 = PH @ theta[2]
    return l11, l21, l22

def M_entries(theta, X):
    """M = ALPHA*I + L^T L for lower-triangular L. Returns (p, q, r) with
    M = [[p, q], [q, r]], plus Mdot entries, all (N,)."""
    PH = feats(X)
    l11, l21, l22 = L_of(theta, PH)
    p = ALPHA + l11 ** 2 + l21 ** 2
    q = l21 * l22
    r = ALPHA + l22 ** 2
    # derivatives of L entries w.r.t. x_k are constants theta[:, 1+k-1]... via DPHI
    F = f(X)
    dp = 0.0; dq = 0.0; dr = 0.0
    for k in range(2):
        dl11 = theta[0] @ DPHI[k]
        dl21 = theta[1] @ DPHI[k]
        dl22 = theta[2] @ DPHI[k]
        dp_k = 2 * l11 * dl11 + 2 * l21 * dl21
        dq_k = dl21 * l22 + l21 * dl22
        dr_k = 2 * l22 * dl22
        dp = dp + dp_k * F[:, k]
        dq = dq + dq_k * F[:, k]
        dr = dr + dr_k * F[:, k]
    return p, q, r, dp, dq, dr

def lam_max_C(theta, X):
    """lambda_max of C(x) = Mdot + A^T M + M A + 2*RATE*M, vectorized, exact
    for symmetric 2x2: lmax = m + sqrt(d^2 + c^2)."""
    p, q, r, dp, dq, dr = M_entries(theta, X)
    A = jac(X)
    a11 = A[:, 0, 0]; a12 = A[:, 0, 1]; a21 = A[:, 1, 0]; a22 = A[:, 1, 1]
    # S = A^T M + M A  (symmetric)
    s11 = 2 * (a11 * p + a21 * q)
    s22 = 2 * (a12 * q + a22 * r)
    s12 = a11 * q + a21 * r + a12 * p + a22 * q
    c11 = dp + s11 + 2 * RATE * p
    c22 = dr + s22 + 2 * RATE * r
    c12 = dq + s12 + 2 * RATE * q
    m = 0.5 * (c11 + c22)
    d = 0.5 * (c11 - c22)
    return m + np.sqrt(d ** 2 + c12 ** 2 + 1e-300)

def lam_max_C_norm(theta, X):
    """Scale-invariant test function g_n(x) = lambda_max(C(x)) / tr M(x).
    Sign-equivalent to the contraction condition since tr M > 0."""
    p, q, r, dp, dq, dr = M_entries(theta, X)
    A = jac(X)
    a11 = A[:, 0, 0]; a12 = A[:, 0, 1]; a21 = A[:, 1, 0]; a22 = A[:, 1, 1]
    s11 = 2 * (a11 * p + a21 * q)
    s22 = 2 * (a12 * q + a22 * r)
    s12 = a11 * q + a21 * r + a12 * p + a22 * q
    c11 = dp + s11 + 2 * RATE * p
    c22 = dr + s22 + 2 * RATE * r
    c12 = dq + s12 + 2 * RATE * q
    m = 0.5 * (c11 + c22)
    d = 0.5 * (c11 - c22)
    return (m + np.sqrt(d ** 2 + c12 ** 2 + 1e-300)) / (p + r)

# ---------------- learner ----------------
def loss(theta, X, eps, normalized=False):
    g = lam_max_C_norm(theta, X) if normalized else lam_max_C(theta, X)
    return np.mean(np.maximum(0.0, g + eps))

dloss = grad(loss)

def adam(theta, X, eps, steps=400, lr=2e-2, normalized=False):
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    b1, b2, e = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        g = dloss(theta, X, eps, normalized)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        theta = theta - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + e)
        if loss(theta, X, eps, normalized) == 0.0:
            break
    return theta

# ---------------- verifier ----------------
def grid(n):
    s = np.linspace(-BOX, BOX, n)
    XX, YY = np.meshgrid(s, s)
    return np.stack([XX.ravel(), YY.ravel()], axis=1), s

def lipschitz_estimate(theta, n=400, normalized=False):
    """Empirical sup-norm of grad_x of the test function, scaled by a
    safety factor. NOT SOUND and diagnostic only: it never enters a
    certificate. Certification uses bounds.lipschitz_certified, which is
    the default path in verify(); this estimator is retained solely as a
    comparison baseline for the conservatism of the certified bound."""
    fn = lam_max_C_norm if normalized else lam_max_C
    X, _ = grid(n)
    h = 1e-5
    g0 = fn(theta, X)
    gx = (fn(theta, X + np.array([h, 0.0])) - g0) / h
    gy = (fn(theta, X + np.array([0.0, h])) - g0) / h
    return LIP_SAFETY * float(np.max(np.sqrt(gx ** 2 + gy ** 2)))

def grid_enclosure(theta, n_grid, normalized=False, P=None, X=None,
                   values=None, block_points=131072):
    """Outward upper grid values for the exact-real stored candidate.

    This function always checks the full grid. The learner's ordinary
    floating-point calculations are independent diagnostic calculations.
    """
    import numpy as onp
    import bounds
    from rounding import grid_radius
    if n_grid < 2 or block_points < 1:
        raise ValueError("invalid grid or block size")
    nodes = onp.linspace(-BOX, BOX, n_grid)
    if nodes[0] != -BOX or nodes[-1] != BOX:
        raise ValueError("grid does not cover the stated box")
    tau = grid_radius(nodes, dimension=2)
    P = bounds.assemble_exact(theta) if P is None else P
    total = n_grid ** 2
    if values is not None and len(values) != total:
        raise ValueError("wrong grid-value storage size")
    if X is not None and X.shape != (total, 2):
        raise ValueError("wrong grid point shape")
    worst = -onp.inf
    for start in range(0, total, block_points):
        stop = min(total, start + block_points)
        if X is None:
            indices = onp.arange(start, stop)
            points = onp.stack([nodes[indices % n_grid], nodes[indices // n_grid]], axis=1)
        else:
            points = onp.asarray(X[start:stop])
            indices = onp.arange(start, stop)
            if (not onp.array_equal(points[:, 0], nodes[indices % n_grid])
                    or not onp.array_equal(points[:, 1], nodes[indices // n_grid])):
                raise ValueError("supplied points do not match the stated Cartesian grid")
        upper = bounds.contraction_grid_interval(theta, points, normalized=normalized, P=P).hi
        if not onp.isfinite(upper).all():
            raise ValueError("nonfinite outward polynomial grid enclosure")
        worst = max(worst, float(onp.max(upper)))
        if values is not None:
            values[start:stop] = upper
    return {"worst_upper": float(worst), "tau": tau, "grid_enclosed": True,
            "outward_rounded": True, "rounding_mode": "outward"}


def verify(theta, n_grid, normalized=False, certified=True, cert_depth=3):
    """Grid oracle with outward-rounded positive answers.

    A nominal evaluation may reject a candidate and propose training points.
    It can never certify: every prospective positive answer additionally
    encloses all grid values using exact coefficients and interval Horner
    evaluation, with an outward covering radius and Lipschitz product.
    The explicit certified=False option remains an empirical diagnostic.
    """
    from rounding import grid_radius, mul_up, sub_down
    if n_grid < 2:
        raise ValueError("n_grid must be >= 2")
    X, s = grid(n_grid)
    if s[0] != -BOX or s[-1] != BOX:
        raise ValueError("grid does not cover the stated box")
    tau_cov = grid_radius(s, dimension=2)
    if certified:
        from bounds import lipschitz_certified
        Lhat = lipschitz_certified(theta, depth=cert_depth, normalized=normalized)
    else:
        Lhat = lipschitz_estimate(theta, normalized=normalized)
    if not np.isfinite(Lhat) or Lhat < 0:
        raise ValueError("invalid Lipschitz bound")
    required = mul_up(float(Lhat), tau_cov) if certified else float(Lhat) * tau_cov
    margin = -required
    g = (lam_max_C_norm if normalized else lam_max_C)(theta, X)
    if not np.all(np.isfinite(g)):
        raise ValueError("nonfinite polynomial test value")
    nominal_worst = float(np.max(g))
    viol = g > margin
    grid_enclosed = False
    if certified and not viol.any():
        # A screening pass is only a request for a rigorous full-grid check.
        grid_enclosure(theta, n_grid, normalized=normalized, X=X, values=g)
        grid_enclosed = True
        viol = g > margin
    worst = float(np.max(g))
    slack = sub_down(-worst, required) if certified else -worst - required
    accepted = bool(not viol.any())
    if certified:
        accepted = bool(grid_enclosed and accepted and slack >= 0.0)
        if not accepted and not viol.any():
            # An exact zero may round to a negative lower slack. Enlarge one
            # valid upper endpoint and report the boundary as inconclusive.
            index = int(np.argmax(g))
            g[index] = np.nextafter(margin, np.inf)
            viol = g > margin
            worst = float(np.max(g))
            slack = sub_down(-worst, required)
    return {
        "certified": accepted,
        "X": X, "g": g, "viol": viol,
        "margin": margin, "L": Lhat, "tau": float(tau_cov),
        "worst": worst, "nominal_worst": nominal_worst,
        "uniform_slack": float(slack), "slack_is_certified": grid_enclosed,
        "grid_enclosed": grid_enclosed, "outward_rounded": bool(certified),
        "rounding_mode": "outward" if certified else "empirical",
        "grid_values_kind": "outward_upper" if grid_enclosed else "nominal_rejection_screen",
    }

# ---------------- CEGIS loop ----------------
def perturb(Xc, m, sigma=0.08):
    reps = np.repeat(Xc, m, axis=0)
    return np.clip(reps + sigma * npr.randn(*reps.shape), -BOX, BOX)

def run(n_grid=201, max_iter=30, k_worst=60, m_pert=5, seed=0):
    npr.seed(seed)
    D = npr.uniform(-BOX, BOX, size=(500, 2))
    theta = 0.1 * npr.randn(3, 3)
    theta[0, 0] = 1.0; theta[2, 0] = 1.0   # init near identity-like metric
    history = []
    eps = EPS
    for it in range(max_iter):
        theta = adam(theta, D, eps, steps=500 if it == 0 else 300)
        V = verify(theta, n_grid)
        # margin coupling: learner must train past the oracle's margin
        eps = max(EPS, 1.3 * V["L"] * V["tau"])
        n_viol = int(V["viol"].sum())
        history.append({"iter": it, "violations": n_viol,
                        "worst": V["worst"], "margin": V["margin"],
                        "L": V["L"], "train_loss": float(loss(theta, D, eps))})
        print(f"[{it:02d}] viol={n_viol:6d}  worst g={V['worst']:+.4f}  "
              f"required<{V['margin']:+.4f}  L={V['L']:.2f}  |D|={len(D)}")
        if V["certified"]:
            print("CERTIFIED")
            return theta, history, V, D, True
        idx = np.argsort(-V["g"])[:k_worst]
        Xc = V["X"][idx]
        D = np.vstack([D, Xc, perturb(Xc, m_pert)])
    return theta, history, V, D, False

if __name__ == "__main__":
    t0 = time.time()
    theta, hist, V, D, ok = run()
    print(f"\nelapsed {time.time()-t0:.1f}s, certified={ok}")
    print("final theta (rows: L11, L21, L22 coeffs on [1, x1, x2]):")
    print(np.round(theta, 3))
    import os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _results = _os.path.join(_here, "results")
    _os.makedirs(_results, exist_ok=True)
    np.save(_os.path.join(_results, "theta.npy"), theta)
    json.dump(hist, open(_os.path.join(_results, "history.json"), "w"))
