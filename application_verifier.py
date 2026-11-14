"""Outward adaptive certificates for the published two-machine swing model.

The physical model is f=(omega, -.5*omega-sin(delta+pi/3)+sin(pi/3)).
The identity sin(delta+pi/3)-sin(pi/3) = .5*sin(delta) +
sqrt(3)/2*(cos(delta)-1) avoids treating a rounded pi/3 as the model.
Stored binary64 metric parameters, rate, floor, domain, and any quadratic
region coefficients denote exact real numbers. Every successful box test uses
outward binary64 arithmetic and directed MPFR transcendental enclosures.

The three factor entries (L11,L21,L22) come from the shared six-unit tanh
network in bench_neural, with 39 stored parameters, and M=alpha*I+L.T*L.
Nominal evaluations only screen rejection and guide
learning. They never provide a positive certificate.
"""
from fractions import Fraction
import hashlib
import json
import math
import time
from functools import lru_cache

import autograd.numpy as anp
import numpy as np

import rounding
import bench_neural
from rounding import Interval as Iv


MODEL_ID = "two_machine_swing_exact_pi_over_3_v1"
PARTITION_RULE = "longest absolute edge, first coordinate breaks ties"


def _parameters(theta, alpha, rate):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (39,) or not np.isfinite(theta).all():
        raise ValueError("finite 39-parameter neural factor required")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("positive finite metric floor required")
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("nonnegative finite target rate required")
    return theta


def _domain(lower, upper):
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if lo.shape != (2,) or hi.shape != (2,) or not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
        raise ValueError("finite nonempty planar box required")
    return lo, hi


def _region(region):
    if region is None:
        return None
    P = np.asarray(region["matrix"], float)
    rho = float(region["level"])
    if P.shape != (2, 2) or not np.isfinite(P).all() or not np.array_equal(P, P.T):
        raise ValueError("finite symmetric quadratic region required")
    # Region exclusion is sound for any symmetric P. Positive definiteness is
    # not assumed here; callers who interpret it as an ellipsoid must prove it.
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError("positive finite quadratic region level required")
    return {"matrix": P.tolist(), "level": rho}


def dynamics_nominal(X):
    """Autograd-compatible nominal dynamics, never a certificate."""
    x, v = X[:, 0], X[:, 1]
    force = .5 * anp.sin(x) + (anp.sqrt(3.) / 2.) * (anp.cos(x) - 1.)
    return anp.stack((v, -.5 * v - force), axis=1)


def contraction_entries(theta, X, rate=.05, alpha=.1):
    """Autograd-compatible c11,c12,c22,trace(M) for the shared neural factor."""
    (a, b, c), dL = bench_neural.L_entries(theta, X)
    p, q, r = alpha + a*a + b*b, b*c, alpha + c*c
    f = dynamics_nominal(X)
    adot = f[:, 0]*dL[0][0] + f[:, 1]*dL[1][0]
    bdot = f[:, 0]*dL[0][1] + f[:, 1]*dL[1][1]
    cdot = f[:, 0]*dL[0][2] + f[:, 1]*dL[1][2]
    pdot, qdot, rdot = 2*a*adot + 2*b*bdot, bdot*c+b*cdot, 2*c*cdot
    k = -.5*anp.cos(X[:, 0]) + (anp.sqrt(3.)/2.)*anp.sin(X[:, 0])
    c11 = pdot + 2*k*q + 2*rate*p
    c12 = qdot + p - .5*q + k*r + 2*rate*q
    c22 = rdot + 2*q - r + 2*rate*r
    return c11, c12, c22, p+r


def contraction_nominal(theta, X, rate=.05, alpha=.1, normalized=False):
    """Nominal maximum eigenvalue for the same factor and physical model."""
    c11, c12, c22, trace = contraction_entries(theta, X, rate, alpha)
    eig = .5*(c11+c22) + anp.sqrt((.5*(c11-c22))**2+c12*c12+1e-300)
    return eig/trace if normalized else eig


def metric_interval(theta, lower, upper, alpha=.1):
    """Enclose M and its two spatial derivatives on the full box."""
    theta = _parameters(theta, alpha, 0.)
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if lo.shape != (2,) or hi.shape != (2,) or not np.isfinite([lo, hi]).all() or np.any(lo > hi):
        raise ValueError("finite closed planar box required")
    x, v = Iv(lo[0], hi[0]), Iv(lo[1], hi[1])
    (a, b, c), dL, _ = bench_neural._iv_network_all(theta, x, v, second=False)
    p, q, r = alpha+a.square()+b.square(), b*c, alpha+c.square()
    deriv = []
    for j in (0, 1):
        da, db, dc = dL[j]
        deriv.append((2*a*da+2*b*db, db*c+b*dc, 2*c*dc))
    return (p, q, r), deriv


@lru_cache(maxsize=1)
def _sqrt3_interval():
    return Iv(rounding.sqrt_down(3.), rounding.sqrt_up(3.)) * .5


def _contraction_batch(theta, lower, upper, rate=.05, alpha=.1):
    """Vectorized interval matrices for rows of lower/upper boxes."""
    x, v = Iv(lower[:, 0], upper[:, 0]), Iv(lower[:, 1], upper[:, 1])
    (a, b, c), dL, _ = bench_neural._iv_network_all(theta, x, v, second=False)
    p, q, r = alpha+a.square()+b.square(), b*c, alpha+c.square()
    deriv = []
    for da, db, dc in dL:
        deriv.append((2*a*da+2*b*db, db*c+b*dc, 2*c*dc))
    half_sqrt3 = _sqrt3_interval()
    sine, cosine = x.sin(), x.cos()
    f2 = -.5*v - .5*sine - half_sqrt3*(cosine-1.)
    k = -.5*cosine + half_sqrt3*sine
    pdot, qdot, rdot = (deriv[0][j]*v + deriv[1][j]*f2 for j in range(3))
    c11 = pdot + 2*k*q + 2*rate*p
    c12 = qdot + p - .5*q + k*r + 2*rate*q
    c22 = rdot + 2*q - r + 2*rate*r
    return c11, c12, c22


def _hull_eigen_upper(c11, c12, c22):
    """Upper lambda_max for every symmetric 2x2 matrix in this interval hull.

    lambda_max is increasing in each diagonal, and in the absolute off-diagonal
    with fixed diagonal. Substituting upper diagonals and max |offdiag| into
    the exact eigenvalue formula therefore gives a rigorous scalar upper bound.
    All arithmetic in that formula is outward rounded.
    """
    p, q, r = Iv(c11.hi), Iv(c12.absmax()), Iv(c22.hi)
    return ((p+r)*.5 + (((p-r)*.5).square()+q.square()).sqrt()).hi


def contraction_interval(theta, lower, upper, rate=.05, alpha=.1):
    """Enclose the exact contraction matrix on an entire closed box."""
    theta = _parameters(theta, alpha, rate)
    (p, q, r), deriv = metric_interval(theta, lower, upper, alpha)
    x, v = Iv(lower[0], upper[0]), Iv(lower[1], upper[1])
    half_sqrt3 = _sqrt3_interval()
    sine, cosine = x.sin(), x.cos()
    f2 = -.5*v - .5*sine - half_sqrt3*(cosine-1.)
    k = -.5*cosine + half_sqrt3*sine
    pdot, qdot, rdot = (deriv[0][j]*v + deriv[1][j]*f2 for j in range(3))
    c11 = pdot + 2*k*q + 2*rate*p
    c12 = qdot + p - .5*q + k*r + 2*rate*q
    c22 = rdot + 2*q - r + 2*rate*r
    return [[c11, c12], [c12, c22]]


def metric_guarantees(theta, lower, upper, rate, margin, alpha=.1):
    """Global metric bounds and downward rate, conservatively on the whole box."""
    theta = _parameters(theta, alpha, rate)
    lo, hi = _domain(lower, upper)
    x, v = Iv(lo[0], hi[0]), Iv(lo[1], hi[1])
    factors, _, _ = bench_neural._iv_network_all(theta, x, v, second=False)
    bound = Iv(alpha)
    for a in factors:
        bound = bound + a.square()
    b = float(bound.hi)
    eta = float(margin)
    if not math.isfinite(eta) or eta < 0:
        raise ValueError("finite nonnegative margin required")
    improved = rounding.add_down(rate, rounding.div_down(eta, rounding.mul_up(2., b))) if eta else rate
    return {"a": alpha, "b": b, "raw_margin_lower": eta,
            "rate_lower": improved,
            "overshoot_upper": rounding.sqrt_up(rounding.div_up(b, alpha)),
            "outward_rounded": True}


def certified_energy_level(lower, upper):
    """Return an inward energy level for a forward-invariant physical subset.

    For symmetric [-B,B]^2 with 0<B<=3/4, U'(x) has the sign of x:
    U'(x)=2 sin(x/2) cos(pi/3+x/2), and |x|<pi/3. Hence U>=0,
    is increasing away from zero, and the minimum energy on the box boundary
    is min(U(-B),U(B),B^2/2). For E=v^2/2+U, E_dot=-v^2/2.
    The in-box sublevel {x in [-B,B]^2: E(x)<=rho} below that boundary
    is forward invariant. This is not the unrestricted global sublevel of
    the tilted potential. Its radial monotonicity makes it star-shaped about zero. On B<=1/2,
    U''=cos(x+pi/3)>0, so this sublevel is convex.
    """
    lo, hi = _domain(lower, upper)
    B = float(hi[0])
    if (not np.array_equal(hi, np.asarray([B, B]))
            or not np.array_equal(lo, -hi) or not 0 < B <= .75):
        raise ValueError("energy certificate requires symmetric box with 0<B<=3/4")
    def potential(x):
        x = Iv(x)
        return .5*(1.-x.cos()) + _sqrt3_interval()*(x.sin()-x)
    e_left, e_right, e_velocity = potential(-B), potential(B), Iv(B).square()*.5
    boundary = min(float(e_left.lo), float(e_right.lo), float(e_velocity.lo))
    factor = Fraction(9, 10)
    rho = float((Iv(rounding.float_down(factor), rounding.float_up(factor))*boundary).lo)
    if not rho > 0:
        raise ArithmeticError("unable to establish a positive inward energy level")
    return {"box_radius": B, "rho_lower": rho,
            "boundary_energy_lower": boundary,
            "potential_left_lower": float(e_left.lo),
            "potential_right_lower": float(e_right.lo),
            "velocity_boundary_lower": float(e_velocity.lo),
            "level_fraction_exact": "9/10", "outward_rounded": True,
            "set_definition": "{(delta,omega) in [-B,B]^2: omega^2/2+U(delta)<=rho_lower}",
            "forward_invariant": True, "star_shaped": True,
            "convex": B <= .5,
            "proof": "E_dot=-omega^2/2; restoring force sign; level strictly below the entire box boundary"}


def _outside_region(lower, upper, region):
    if region is None:
        return False
    x, v = Iv(lower[0], upper[0]), Iv(lower[1], upper[1])
    P = region["matrix"]
    q = P[0][0]*x.square() + 2*P[0][1]*x*v + P[1][1]*v.square()
    return float(q.lo) > region["level"]


def _inside_nominal(points, region):
    if region is None:
        return np.ones(len(points), dtype=bool)
    P = np.asarray(region["matrix"])
    return np.einsum("ni,ij,nj->n", points, P, points) <= region["level"]


def candidate_hash(theta, rate, alpha):
    data = {"model": MODEL_ID, "theta": np.asarray(theta, float).tolist(),
            "rate": rate, "alpha": alpha}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _split(lo, hi):
    axis = int(np.argmax(hi-lo))
    midpoint = lo[axis] + (hi[axis]-lo[axis])/2
    if not lo[axis] < midpoint < hi[axis]:
        raise ArithmeticError("unrepresentable interior split")
    lh, rl = hi.copy(), lo.copy()
    lh[axis] = midpoint
    rl[axis] = midpoint
    return (lo.copy(), lh), (rl, hi.copy())


def adaptive_oracle(theta, lower, upper, *, rate=.05, alpha=.1,
                    target_margin=1e-4, max_boxes=12000, max_depth=24,
                    check_points=None, quadratic_region=None, max_diagnostics=64):
    """Certify every leaf, or return a fail-closed inconclusive outcome.

    A quadratic_region restricts the verification set to the intersection of
    the box and x.T P x <= level. An excluded box needs a strict outward lower
    bound above level. Boundary-intersecting boxes require a full-box matrix
    proof, so no boundary omission can produce a positive answer.
    """
    start = time.perf_counter()
    theta = _parameters(theta, alpha, rate)
    lo0, hi0 = _domain(lower, upper)
    region = _region(quadratic_region)
    if (type(max_boxes) is not int or max_boxes < 1 or type(max_depth) is not int
            or max_depth < 0 or type(max_diagnostics) is not int or max_diagnostics < 1
            or not math.isfinite(target_margin) or target_margin < 0):
        raise ValueError("invalid adaptive verification budget")
    points = np.asarray(check_points, float) if check_points is not None else np.empty((0, 2))
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or np.any(points < lo0) or np.any(points > hi0)):
        raise ValueError("diagnostic points must lie in the verification box")
    points = points[_inside_nominal(points, region)]
    counters = {"boxes_evaluated": 0, "accepted_leaves": 0, "excluded_leaves": 0,
                "splits": 0, "maximum_depth_reached": 0,
                "nominal_point_evaluations": len(points)}
    queue = [(lo0, hi0, "")]
    accepted, excluded, diagnostics = [], [], []
    reason, sampled_worst = "all_leaves_certified", None
    try:
        vals = np.asarray(contraction_nominal(theta, points, rate, alpha)) if len(points) else np.empty(0)
        if not np.isfinite(vals).all():
            raise ArithmeticError("nonfinite nominal screen")
        if len(vals):
            sampled_worst = float(np.max(vals))
            if sampled_worst >= -target_margin:
                diagnostics = points[np.argsort(-vals)[:max_diagnostics]].tolist()
                queue = []
                reason = "nominal_screen_inconclusive"
        while queue:
            if counters["boxes_evaluated"] >= max_boxes:
                reason = "box_budget_exhausted"
                diagnostics = [(lo+(hi-lo)/2).tolist() for lo, hi, _ in queue[-max_diagnostics:]]
                break
            batch = [queue.pop() for _ in range(min(128, len(queue), max_boxes-counters["boxes_evaluated"]))]
            lower_batch = np.stack([q[0] for q in batch])
            upper_batch = np.stack([q[1] for q in batch])
            upper_eigen = _hull_eigen_upper(*_contraction_batch(theta, lower_batch, upper_batch, rate, alpha))
            counters["boxes_evaluated"] += len(batch)
            counters["maximum_depth_reached"] = max(counters["maximum_depth_reached"], max(len(q[2]) for q in batch))
            for index, (lo, hi, path) in enumerate(batch):
                if _outside_region(lo, hi, region):
                    excluded.append(path)
                    counters["excluded_leaves"] += 1
                elif upper_eigen[index] <= -target_margin:
                    accepted.append(path)
                    counters["accepted_leaves"] += 1
                else:
                    if len(path) >= max_depth:
                        reason = "depth_budget_exhausted"
                        diagnostics = [(lo+(hi-lo)/2).tolist()]
                        break
                    try:
                        left, right = _split(lo, hi)
                    except ArithmeticError:
                        reason = "resolution_exhausted"
                        diagnostics = [(lo+(hi-lo)/2).tolist()]
                        break
                    queue.append((*right, path+"1"))
                    queue.append((*left, path+"0"))
                    counters["splits"] += 1
            if reason != "all_leaves_certified":
                break
    except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError, np.linalg.LinAlgError):
        reason = "arithmetic_inconclusive"
        diagnostics = [(lo0+(hi0-lo0)/2).tolist()]
    certified = reason == "all_leaves_certified" and not queue and not diagnostics
    return {"model": MODEL_ID, "status": "certified" if certified else "inconclusive",
            "certified": certified, "reason": reason, "outward_rounded": True,
            "domain_lower": lo0.tolist(), "domain_upper": hi0.tolist(),
            "quadratic_region": region, "rate": rate, "alpha": alpha,
            "target_margin": target_margin,
            "uniform_raw_margin_lower": target_margin if certified else None,
            "budget": {"max_boxes": max_boxes, "max_depth": max_depth},
            "counters": counters, "training_points": diagnostics,
            "diagnostic_points": diagnostics,
            "sampled_worst_diagnostic": sampled_worst,
            "proof": {"candidate_sha256": candidate_hash(theta, rate, alpha),
                      "leaf_paths": accepted if certified else [],
                      "excluded_paths": excluded if certified else [],
                      "partition_rule": PARTITION_RULE},
            "positive_decision": "outward interval contraction matrix and outward worst-case symmetric 2-by-2 hull eigenvalue on each nonexcluded partition leaf",
            "wall_s": time.perf_counter()-start}


def replay_oracle_certificate(result, theta):
    """Reconstruct a full partition and recheck every positive leaf outward."""
    if result.get("model") != MODEL_ID or result.get("outward_rounded") is not True:
        raise ValueError("unrecognized or unrounded swing certificate")
    if result.get("status") not in ("certified", "inconclusive") or result.get("certified") != (result["status"] == "certified"):
        raise ValueError("inconsistent certificate status")
    rate, alpha = result["rate"], result["alpha"]
    theta = _parameters(theta, alpha, rate)
    lo0, hi0 = _domain(result["domain_lower"], result["domain_upper"])
    region = _region(result.get("quadratic_region"))
    margin, budget, counters = result["target_margin"], result["budget"], result["counters"]
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("invalid certificate margin")
    if (type(budget.get("max_boxes")) is not int or budget["max_boxes"] < 1
            or type(budget.get("max_depth")) is not int or budget["max_depth"] < 0
            or any(type(counters.get(k)) is not int or counters[k] < 0 for k in
                   ("boxes_evaluated", "accepted_leaves", "excluded_leaves", "splits", "maximum_depth_reached", "nominal_point_evaluations"))
            or counters["boxes_evaluated"] > budget["max_boxes"]
            or counters["maximum_depth_reached"] > budget["max_depth"]):
        raise ValueError("invalid certificate work counters")
    proof = result["proof"]
    if proof.get("candidate_sha256") != candidate_hash(theta, rate, alpha) or proof.get("partition_rule") != PARTITION_RULE:
        raise ValueError("certificate candidate mismatch")
    if not result["certified"]:
        if (result.get("uniform_raw_margin_lower") is not None or proof.get("leaf_paths") != []
                or proof.get("excluded_paths") != [] or not result.get("training_points")):
            raise ValueError("inconclusive result claims proof or lacks diagnostics")
        return 0
    if result.get("reason") != "all_leaves_certified" or result.get("uniform_raw_margin_lower") != margin:
        raise ValueError("invalid positive margin")
    accepted, excluded = proof.get("leaf_paths"), proof.get("excluded_paths")
    if not isinstance(accepted, list) or not isinstance(excluded, list):
        raise ValueError("missing partition")
    paths = accepted + excluded
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("empty or duplicate partition")
    if any(not isinstance(p, str) or any(c not in "01" for c in p) or len(p) > budget["max_depth"] for p in paths):
        raise ValueError("invalid partition path")
    ordered = sorted(paths)
    if any(b.startswith(a) for a, b in zip(ordered, ordered[1:])):
        raise ValueError("overlapping partition leaves")
    if sum((Fraction(1, 2**len(p)) for p in paths), Fraction(0)) != 1:
        raise ValueError("incomplete partition cover")
    n = len(paths)
    if (counters["accepted_leaves"] != len(accepted) or counters["excluded_leaves"] != len(excluded)
            or counters["splits"] != n-1 or counters["boxes_evaluated"] != 2*n-1
            or counters["maximum_depth_reached"] != max(map(len, paths))):
        raise ValueError("inconsistent positive partition work")
    accepted, excluded = set(accepted), set(excluded)
    stack = [(lo0, hi0, "")]
    accepted_boxes = []
    while stack:
        lo, hi, path = stack.pop()
        if path in accepted:
            accepted_boxes.append((lo, hi))
        elif path in excluded:
            if not _outside_region(lo, hi, region):
                raise ValueError("excluded leaf intersects verification region")
        else:
            left, right = _split(lo, hi)
            stack.append((*right, path+"1"))
            stack.append((*left, path+"0"))
    for begin in range(0, len(accepted_boxes), 128):
        batch = accepted_boxes[begin:begin+128]
        lo, hi = np.stack([q[0] for q in batch]), np.stack([q[1] for q in batch])
        if np.any(_hull_eigen_upper(*_contraction_batch(theta, lo, hi, rate, alpha)) > -margin):
            raise ValueError("positive leaf failed outward replay")
    return n
