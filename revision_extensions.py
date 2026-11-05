"""Certified research extensions: adaptive boxes, a coupled 3D loop, and fitted dynamics.

Every positive answer is a polynomial identity followed by outward interval LDL^T
inequalities. NumPy eigendecompositions serve nominal diagnostics. The adaptive
oracle is sufficient, not complete at finite budget, and remains exponential in
the worst case. The learned-dynamics example uses known polynomial truth to bound
the residual globally; trajectory fitting alone does not supply those bounds.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import autograd.numpy as anp
from autograd import grad
import numpy as np

from rounding import (Interval as Iv, float_down, float_up, add_up, mul_up,
                      sub_down, div_down, div_up, sqrt_up, backend_info, self_test,
                      grid_radius)

ROOT = Path(__file__).resolve().parent


def rational(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    value = float(value)
    if not math.isfinite(value):
        raise ArithmeticError("nonfinite exact polynomial input")
    return Fraction.from_float(value)


class ExactPoly:
    """Sparse, general-dimensional polynomials over exact rational coefficients."""

    def __init__(self, n, coeffs=None):
        self.n = int(n)
        if self.n < 1:
            raise ValueError("positive dimension required")
        self.coeffs = {}
        for powers, value in (coeffs or {}).items():
            powers = tuple(powers)
            if len(powers) != n or any(not isinstance(k, int) or k < 0 for k in powers):
                raise ValueError("invalid polynomial exponent")
            value = rational(value)
            if value:
                self.coeffs[powers] = value

    @classmethod
    def constant(cls, n, value):
        return cls(n, {(0,) * n: value})

    @classmethod
    def coordinate(cls, n, k):
        powers = [0] * n
        powers[k] = 1
        return cls(n, {tuple(powers): 1})

    def _coerce(self, other):
        if isinstance(other, ExactPoly):
            if other.n != self.n:
                raise ValueError("polynomial dimension mismatch")
            return other
        return ExactPoly.constant(self.n, other)

    def __add__(self, other):
        other = self._coerce(other)
        out = dict(self.coeffs)
        for k, v in other.coeffs.items():
            out[k] = out.get(k, Fraction(0)) + v
        return ExactPoly(self.n, out)

    __radd__ = __add__

    def __neg__(self):
        return ExactPoly(self.n, {k: -v for k, v in self.coeffs.items()})

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        out = {}
        for a, u in self.coeffs.items():
            for b, v in other.coeffs.items():
                k = tuple(x + y for x, y in zip(a, b))
                out[k] = out.get(k, Fraction(0)) + u * v
        return ExactPoly(self.n, out)

    __rmul__ = __mul__

    def __pow__(self, power):
        if not isinstance(power, int) or power < 0:
            raise ValueError("nonnegative integer power required")
        out = ExactPoly.constant(self.n, 1)
        for _ in range(power):
            out = out * self
        return out

    def derivative(self, k):
        out = {}
        for powers, value in self.coeffs.items():
            if powers[k]:
                q = list(powers)
                q[k] -= 1
                out[tuple(q)] = value * powers[k]
        return ExactPoly(self.n, out)

    def exact_value(self, point):
        point = tuple(map(rational, point))
        if len(point) != self.n:
            raise ValueError("point dimension mismatch")
        return sum((v * math.prod(x ** k for x, k in zip(point, powers))
                    for powers, v in self.coeffs.items()), Fraction(0))

    def evaluate(self, points, xp=np):
        points = xp.asarray(points) if xp is np else points
        out = xp.zeros(points.shape[0])
        for powers, value in self.coeffs.items():
            term = xp.ones(points.shape[0]) * float(value)
            for k, power in enumerate(powers):
                if power:
                    term = term * points[:, k] ** power
            out = out + term
        return out

    def shifted(self, center):
        """Binomial shift over Q, with no rounded intermediate coefficients."""
        center = tuple(map(rational, center))
        if len(center) != self.n:
            raise ValueError("center dimension mismatch")
        out = {}
        for powers, value in self.coeffs.items():
            for beta in product(*(range(k + 1) for k in powers)):
                v = value
                for j, (k, b) in enumerate(zip(powers, beta)):
                    v *= math.comb(k, b) * center[j] ** (k - b)
                if v:
                    out[beta] = out.get(beta, Fraction(0)) + v
        return ExactPoly(self.n, out)

    def enclosure(self, lower, upper):
        """Recenter exactly and enclose all nonconstant terms by their radii."""
        lo, hi = tuple(map(rational, lower)), tuple(map(rational, upper))
        if len(lo) != self.n or len(hi) != self.n or any(a > b for a, b in zip(lo, hi)):
            raise ValueError("invalid polynomial box")
        center = tuple((a + b) / 2 for a, b in zip(lo, hi))
        radius = tuple((b - a) / 2 for a, b in zip(lo, hi))
        shifted = self.shifted(center)
        constant = shifted.coeffs.get((0,) * self.n, Fraction(0))
        remainder = sum((abs(v) * math.prod(r ** k for r, k in zip(radius, powers))
                         for powers, v in shifted.coeffs.items() if any(powers)), Fraction(0))
        return Iv(float_down(constant - remainder), float_up(constant + remainder))

    def serializable(self):
        return [{"powers": list(k), "numerator": str(v.numerator),
                 "denominator": str(v.denominator)}
                for k, v in sorted(self.coeffs.items())]


def polynomial_matrix_contraction(metric, dynamics, rate):
    """Exact C = sum dM f + A^T M + M A + 2 lambda M, for arbitrary n."""
    n = len(dynamics)
    if len(metric) != n or any(len(row) != n for row in metric):
        raise ValueError("matrix dimension mismatch")
    if any(metric[i][j].coeffs != metric[j][i].coeffs for i in range(n) for j in range(n)):
        raise ValueError("metric must be symmetric")
    jacobian = [[dynamics[i].derivative(k) for k in range(n)] for i in range(n)]
    return [[sum((metric[i][j].derivative(k) * dynamics[k]
                  + jacobian[k][i] * metric[k][j]
                  + metric[i][k] * jacobian[k][j] for k in range(n)),
                 ExactPoly.constant(n, 0)) + metric[i][j] * (2 * rational(rate))
             for j in range(n)] for i in range(n)]


def factor_metric(theta, alpha):
    theta = np.asarray(theta, dtype=float)
    n = theta.shape[1] - 1
    pairs = [(i, j) for i in range(n) for j in range(i + 1)]
    if theta.shape != (len(pairs), n + 1):
        raise ValueError("factor coefficient shape mismatch")
    zero = ExactPoly.constant(n, 0)
    features = [ExactPoly.constant(n, 1)] + [ExactPoly.coordinate(n, k) for k in range(n)]
    factor = [[zero for _ in range(n)] for _ in range(n)]
    for row, (i, j) in zip(theta, pairs):
        factor[i][j] = sum((f * v for f, v in zip(features, row)), zero)
    metric = [[sum((factor[k][i] * factor[k][j] for k in range(n)), zero)
               + (alpha if i == j else 0) for j in range(n)] for i in range(n)]
    return metric


def _positive_definite_interval(matrix):
    n = len(matrix)
    L = [[Iv(float(i == j)) for j in range(n)] for i in range(n)]
    D = []
    try:
        for j in range(n):
            pivot = matrix[j][j]
            for k in range(j):
                pivot = pivot - L[j][k].square() * D[k]
            if float(pivot.lo) <= 0:
                return False
            D.append(pivot)
            for i in range(j + 1, n):
                entry = matrix[i][j]
                for k in range(j):
                    entry = entry - L[i][k] * L[j][k] * D[k]
                L[i][j] = entry / pivot
        return True
    except (ArithmeticError, ZeroDivisionError, ValueError):
        return False


def negative_definite_interval(matrix, margin=0.0):
    """Sufficient proof C <= -margin I by outward interval LDL^T.

    If direct LDL fails, a nominal Cholesky inverse selects an upper triangular
    congruence R. Its stored nonzero diagonal proves invertibility exactly, and
    the full transformed interval R^T(-C-margin I)R is checked again. The nominal
    factorization never supplies a positive answer or an eigenvalue bound.
    """
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if float(matrix[i][j].lo) != float(matrix[j][i].lo) or float(matrix[i][j].hi) != float(matrix[j][i].hi):
                raise ValueError("symmetric interval matrix required")
    positive = [[-matrix[i][j] - (margin if i == j else 0.) for j in range(n)] for i in range(n)]
    if _positive_definite_interval(positive):
        return True
    try:
        center = np.asarray([[float(a.lo) + (float(a.hi) - float(a.lo)) / 2 for a in row] for row in positive])
        R = np.triu(np.linalg.inv(np.linalg.cholesky(center)).T)
        if not np.isfinite(R).all() or np.any(np.diag(R) == 0):
            return False
        transformed = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                value = Iv(0.)
                for k in range(i + 1):
                    for l in range(j + 1):
                        value = value + positive[k][l] * R[k, i] * R[l, j]
                transformed[i][j] = transformed[j][i] = value
        return _positive_definite_interval(transformed)
    except (ArithmeticError, ValueError, np.linalg.LinAlgError, ZeroDivisionError):
        return False


@dataclass(frozen=True)
class OracleBudget:
    max_boxes: int = 12000
    max_depth: int = 24
    target_margin: float = 1e-4


def bernstein_coefficients(poly, lower, upper):
    """Exact initial tensor Bernstein coefficients, then outward conversion.

    On the unit cube the polynomial lies in the convex hull of these
    coefficients. Subsequent de Casteljau averages use outward intervals.
    """
    lo, hi = tuple(map(rational, lower)), tuple(map(rational, upper))
    widths = [b - a for a, b in zip(lo, hi)]
    shifted = poly.shifted(lo)
    power = {k: v * math.prod(w ** j for w, j in zip(widths, k))
             for k, v in shifted.coeffs.items()}
    degrees = [max((k[j] for k in power), default=0) for j in range(poly.n)]
    shape = tuple(d + 1 for d in degrees)
    lower_coeff, upper_coeff = np.empty(shape), np.empty(shape)
    for index in product(*(range(s) for s in shape)):
        value = Fraction(0)
        for powers, coefficient in power.items():
            if all(k <= i for k, i in zip(powers, index)):
                factor = math.prod(Fraction(math.comb(i, k), math.comb(d, k))
                                   for i, d, k in zip(index, degrees, powers))
                value += coefficient * factor
        lower_coeff[index], upper_coeff[index] = float_down(value), float_up(value)
    return Iv(lower_coeff, upper_coeff)


def split_bernstein(coefficients, axis, ratio=Fraction(1, 2)):
    """Outward de Casteljau at the exact stored-endpoint split fraction."""
    ratio = rational(ratio)
    if not 0 < ratio < 1:
        raise ValueError("strict interior Bernstein split required")
    rho = Iv(float_down(ratio), float_up(ratio))
    current = Iv(np.moveaxis(coefficients.lo, axis, 0), np.moveaxis(coefficients.hi, axis, 0))
    degree = current.lo.shape[0] - 1
    if degree == 0:
        return coefficients, coefficients
    left_lo, left_hi = [current.lo[0]], [current.hi[0]]
    right_lo, right_hi = [current.lo[-1]], [current.hi[-1]]
    for _ in range(degree):
        a, b = Iv(current.lo[:-1], current.hi[:-1]), Iv(current.lo[1:], current.hi[1:])
        current = (a + b) * .5 if ratio == Fraction(1, 2) else (1 - rho) * a + rho * b
        left_lo.append(current.lo[0]); left_hi.append(current.hi[0])
        right_lo.append(current.lo[-1]); right_hi.append(current.hi[-1])
    left = Iv(np.moveaxis(np.stack(left_lo), 0, axis), np.moveaxis(np.stack(left_hi), 0, axis))
    right = Iv(np.moveaxis(np.stack(right_lo[::-1]), 0, axis), np.moveaxis(np.stack(right_hi[::-1]), 0, axis))
    return left, right


def _initial_bernstein_matrix(contraction, lower, upper):
    n = len(contraction)
    return [[bernstein_coefficients(contraction[i][j], lower, upper) if j >= i else None
             for j in range(n)] for i in range(n)]


def _bernstein_matrix_enclosure(coefficients):
    n = len(coefficients)
    result = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            c = coefficients[i][j]
            result[i][j] = result[j][i] = Iv(float(np.min(c.lo)), float(np.max(c.hi)))
    return result


def _split_bernstein_matrix(coefficients, axis, ratio):
    n = len(coefficients)
    left, right = [[None] * n for _ in range(n)], [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            left[i][j], right[i][j] = split_bernstein(coefficients[i][j], axis, ratio)
    return left, right


def _split_weights(contraction, lower, upper):
    # These weights are only a subdivision heuristic. No positive answer uses
    # their numerical accuracy. A full prefix partition still covers the box.
    n = len(contraction)
    radius = np.maximum(np.abs(lower), np.abs(upper))
    weights = np.zeros(n)
    for row in contraction:
        for poly in row:
            for powers, coefficient in poly.coeffs.items():
                for j, power in enumerate(powers):
                    if power:
                        remaining = list(powers); remaining[j] -= 1
                        weights[j] += abs(float(coefficient)) * power * math.prod(r ** k for r, k in zip(radius, remaining))
    return np.maximum(weights, 1e-12)


def adaptive_oracle(contraction, lower, upper, budget=OracleBudget(), check_points=None):
    """Partition a box by weighted-edge bisection, certifying every accepted leaf.

    Returned sample points only guide learning. Inconclusive means that the box
    budget, depth, arithmetic, or a diagnostic point prevented a certificate;
    it does not establish a contraction violation.
    """
    start = time.perf_counter()
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    n = len(contraction)
    if lower.shape != (n,) or upper.shape != (n,) or not np.isfinite([lower, upper]).all() or np.any(lower >= upper):
        raise ValueError("nonempty finite box required")
    if (not isinstance(budget.max_boxes, int) or not isinstance(budget.max_depth, int)
            or budget.max_boxes < 1 or budget.max_depth < 0
            or not math.isfinite(budget.target_margin) or budget.target_margin < 0):
        raise ValueError("invalid oracle budget")
    if any(p.n != n for row in contraction for p in row):
        raise ValueError("polynomial dimension mismatch")
    if any(contraction[i][j].coeffs != contraction[j][i].coeffs for i in range(n) for j in range(n)):
        raise ValueError("symmetric polynomial matrix required")
    nominal_points = np.asarray(check_points, float) if check_points is not None else np.empty((0, n))
    if nominal_points.size and (nominal_points.ndim != 2 or nominal_points.shape[1] != n
                               or np.any(nominal_points < lower) or np.any(nominal_points > upper)):
        raise ValueError("diagnostic points outside verification box")
    arithmetic_inconclusive = False
    try:
        initial_nominal = matrix_values(contraction, nominal_points) if nominal_points.size else np.empty((0, n, n))
        if not np.isfinite(initial_nominal).all():
            raise ArithmeticError("nonfinite diagnostic matrix")
        sampled_values = np.linalg.eigvalsh(initial_nominal)[:, -1] if nominal_points.size else np.empty(0)
        split_weights = _split_weights(contraction, lower, upper)
        if not np.isfinite(split_weights).all():
            raise ArithmeticError("nonfinite subdivision heuristic")
    except (ArithmeticError, OverflowError, ValueError, np.linalg.LinAlgError):
        arithmetic_inconclusive = True
        sampled_values = np.empty(0)
        split_weights = np.ones(n)
    sampled_worst = float(np.max(sampled_values)) if len(sampled_values) else None
    counters = {"boxes_evaluated": 0, "accepted_leaves": 0, "splits": 0,
                "maximum_depth_reached": 0, "nominal_point_evaluations": len(nominal_points)}
    queue = [(lower, upper, "", None)]
    accepted_paths = []
    unresolved = []
    reason = "all_leaves_certified"
    # A diagnostic screen may reject early, but never produces a positive answer.
    if arithmetic_inconclusive:
        unresolved = [(lower + (upper - lower) / 2).tolist()]
        reason = "arithmetic_inconclusive"
        queue = []
    elif sampled_worst is not None and sampled_worst >= -budget.target_margin:
        order = np.argsort(-sampled_values)[:60]
        unresolved = nominal_points[order].tolist()
        reason = "nominal_screen_inconclusive"
        queue = []
    else:
        while queue:
            if counters["boxes_evaluated"] >= budget.max_boxes:
                reason = "box_budget_exhausted"
                unresolved.extend(((lo + hi) / 2).tolist() for lo, hi, _, _ in queue[-60:])
                break
            lo, hi, path, coefficients = queue.pop()
            depth = len(path)
            counters["boxes_evaluated"] += 1
            counters["maximum_depth_reached"] = max(depth, counters["maximum_depth_reached"])
            try:
                if coefficients is None:
                    coefficients = _initial_bernstein_matrix(contraction, lo, hi)
                intervals = _bernstein_matrix_enclosure(coefficients)
                passed = negative_definite_interval(intervals, budget.target_margin)
            except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
                reason = "arithmetic_inconclusive"
                unresolved.append(((lo + hi) / 2).tolist())
                break
            if passed:
                counters["accepted_leaves"] += 1
                accepted_paths.append(path)
                continue
            if depth >= budget.max_depth:
                reason = "depth_budget_exhausted"
                unresolved.append(((lo + hi) / 2).tolist())
                break
            axis = int(np.argmax((hi - lo) * split_weights))
            mid = lo[axis] + (hi[axis] - lo[axis]) / 2
            if not lo[axis] < mid < hi[axis]:
                reason = "resolution_exhausted"
                unresolved.append(((lo + hi) / 2).tolist())
                break
            left_hi, right_lo = hi.copy(), lo.copy()
            left_hi[axis] = mid
            right_lo[axis] = mid
            ratio = (rational(mid) - rational(lo[axis])) / (rational(hi[axis]) - rational(lo[axis]))
            left_coeff, right_coeff = _split_bernstein_matrix(coefficients, axis, ratio)
            queue.append((right_lo, hi.copy(), path + "1", right_coeff))
            queue.append((lo.copy(), left_hi, path + "0", left_coeff))
            counters["splits"] += 1
    certified = reason == "all_leaves_certified" and not queue and not unresolved
    return {"status": "certified" if certified else "inconclusive", "certified": certified,
            "reason": reason, "uniform_raw_margin_lower": budget.target_margin if certified else None,
            "domain_lower": lower.tolist(), "domain_upper": upper.tolist(),
            "target_margin": budget.target_margin, "budget": vars(budget),
            "counters": counters, "training_points": unresolved[:60],
            "sampled_worst_diagnostic": sampled_worst, "outward_rounded": True,
            "proof": {"polynomial_sha256": polynomial_matrix_hash(contraction),
                      "leaf_paths": accepted_paths if certified else [],
                      "partition_rule": "coefficient-weighted longest-edge bisection, lowest axis breaks ties",
                      "split_weights": split_weights.tolist()},
            "positive_decision": "exact initial Bernstein coefficients, outward de Casteljau, and outward interval LDLT, optionally after a verified invertible triangular congruence, on every partition leaf",
            "wall_s": time.perf_counter() - start}


def polynomial_matrix_hash(matrix):
    payload = [[p.serializable() for p in row] for row in matrix]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def matrix_values(matrix, points):
    return np.stack([np.stack([p.evaluate(points) for p in row], axis=1)
                     for row in matrix], axis=1)


def benchmark_dynamics(n):
    x = [ExactPoly.coordinate(n, k) for k in range(n)]
    if n == 2:
        return [-x[0], -x[1] + x[0] ** 2]
    if n == 3:
        return [-x[0], -x[1] + x[0] ** 2,
                -x[2] + x[0] * x[1] - x[0] ** 3]
    raise ValueError("declared benchmarks have dimension two or three")


def independent_pullback_identity():
    n = 3
    x = [ExactPoly.coordinate(n, k) for k in range(n)]
    z, one = ExactPoly.constant(n, 0), ExactPoly.constant(n, 1)
    phi = [x[0], x[1] + x[0] ** 2, x[2] + x[0] * x[1]]
    J = [[p.derivative(k) for k in range(n)] for p in phi]
    f = benchmark_dynamics(n)
    conjugacy = [sum((J[i][k] * f[k] for k in range(n)), z) + phi[i] for i in range(n)]
    metric = [[sum((J[k][i] * J[k][j] for k in range(n)), z)
               for j in range(n)] for i in range(n)]
    C = polynomial_matrix_contraction(metric, f, Fraction(2, 5))
    identity = [[C[i][j] + Fraction(6, 5) * metric[i][j]
                 for j in range(n)] for i in range(n)]
    if any(p.coeffs for p in conjugacy) or any(p.coeffs for row in identity for p in row):
        raise AssertionError("independent pullback identity failed")
    # J is unit lower triangular, hence invertible everywhere. The inverse map
    # is (y1, y2-y1^2, y3-y1*(y2-y1^2)), so this is a global diffeomorphism.
    return {"phi": [p.serializable() for p in phi], "dynamics": [p.serializable() for p in f],
            "conjugacy_identity_exact": True, "contraction_identity_exact": True,
            "identity": "C_(lambda=2/5) = -(6/5) J_phi^T J_phi",
            "unit_triangular_jacobian": True, "reference_rate": 1.0,
            "known_metric_used_for_initialization_or_training": False}


def metric_and_contraction_nominal(theta, points, dynamics, alpha, rate):
    """Autograd-compatible actual full factor learner, arbitrary n."""
    n = points.shape[1]
    pairs = [(i, j) for i in range(n) for j in range(i + 1)]
    features = anp.concatenate([anp.ones((len(points), 1)), points], axis=1)
    values = anp.dot(features, theta.T)
    index = {pair: k for k, pair in enumerate(pairs)}
    zeros = anp.zeros(len(points))
    L = anp.stack([anp.stack([values[:, index[(i, j)]] if j <= i else zeros
                              for j in range(n)], axis=1) for i in range(n)], axis=1)
    M = anp.einsum("nki,nkj->nij", L, L) + alpha * anp.eye(n)[None, :, :]
    F = anp.stack([p.evaluate(points, anp) for p in dynamics], axis=1)
    A = anp.stack([anp.stack([p.derivative(k).evaluate(points, anp) for k in range(n)], axis=1)
                   for p in dynamics], axis=1)
    dM = []
    for k in range(n):
        dL = anp.stack([anp.stack([theta[index[(i, j)], k + 1] * anp.ones(len(points))
                                  if j <= i else zeros for j in range(n)], axis=1)
                        for i in range(n)], axis=1)
        dM.append(anp.einsum("nki,nkj->nij", dL, L) + anp.einsum("nki,nkj->nij", L, dL))
    Mdot = sum(dM[k] * F[:, k, None, None] for k in range(n))
    C = (Mdot + anp.einsum("nki,nkj->nij", A, M)
         + anp.einsum("nik,nkj->nij", M, A) + 2 * rate * M)
    return M, C


def train_metric(theta, points, dynamics, alpha, rate, margin, steps, lr):
    """Adam with a nominal largest-eigenvector subgradient for training only."""
    theta = np.asarray(theta, float).copy()
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    started = time.perf_counter()
    final_loss = None
    for step in range(1, steps + 1):
        M, C = metric_and_contraction_nominal(theta, points, dynamics, alpha, rate)
        eigvals, eigvecs = np.linalg.eigh(np.asarray(C))
        top = eigvecs[:, :, -1]
        normalized = eigvals[:, -1] / np.trace(np.asarray(M), axis1=1, axis2=2)
        final_loss = float(np.mean(np.maximum(0, normalized + margin)))
        if final_loss == 0:
            break
        def objective(th):
            metric, contraction = metric_and_contraction_nominal(th, points, dynamics, alpha, rate)
            rayleigh = anp.einsum("ni,nij,nj->n", top, contraction, top)
            trace = sum(metric[:, i, i] for i in range(points.shape[1]))
            return anp.mean(anp.maximum(0, rayleigh / trace + margin))
        gradient = np.asarray(grad(objective)(theta))
        if not np.isfinite(gradient).all():
            raise ArithmeticError("nonfinite learner gradient")
        m = .9 * m + .1 * gradient
        v = .999 * v + .001 * gradient ** 2
        theta = theta - lr * (m / (1 - .9 ** step)) / (np.sqrt(v / (1 - .999 ** step)) + 1e-8)
    return theta, {"steps": step, "final_empirical_loss": final_loss,
                   "wall_s": time.perf_counter() - started, "optimizer_state": "reset each round"}


def diagnostic_grid(n, lower, upper, axis_points=9):
    axes = [np.linspace(a, b, axis_points) for a, b in zip(lower, upper)]
    return np.asarray(list(product(*axes)), float)


def run_learned_metric(dynamics, seed, *, B, alpha=.1, rate=.4, margin=.005,
                       max_rounds=8, steps=300, initial_samples=256, lr=.02,
                       budget=OracleBudget()):
    n = len(dynamics)
    rng = np.random.RandomState(seed)
    pairs = [(i, j) for i in range(n) for j in range(i + 1)]
    theta = rng.normal(0, .01, (len(pairs), n + 1))
    for k, (i, j) in enumerate(pairs):
        if i == j:
            theta[k, 0] += 1
    data = rng.uniform(-B, B, (initial_samples, n))
    screen = diagnostic_grid(n, [-B] * n, [B] * n)
    trace = []
    stop_reason = "round_budget_exhausted"
    for iteration in range(max_rounds + 1):
        exact = factor_metric(theta, alpha)
        contraction = polynomial_matrix_contraction(exact, dynamics, rate)
        verification = adaptive_oracle(contraction, [-B] * n, [B] * n, budget, check_points=screen)
        print(f"ADAPTIVE n={n} seed={seed} round={iteration} status={verification['status']} "
              f"reason={verification['reason']} boxes={verification['counters']['boxes_evaluated']} "
              f"wall={verification['wall_s']:.3f}", flush=True)
        row = {"round": iteration, "oracle": verification, "sample_count": len(data)}
        trace.append(row)
        if verification["certified"] or iteration == max_rounds:
            if verification["certified"]:
                stop_reason = "certified"
            break
        nominated = np.asarray(verification["training_points"], float).reshape(-1, n)
        if not len(nominated):
            raise AssertionError("inconclusive adaptive oracle returned no training point")
        # Add nominal worst screen points as well as unresolved-box centers.
        sampled_C = matrix_values(contraction, screen)
        order = np.argsort(-np.linalg.eigvalsh(sampled_C)[:, -1])[:60]
        nominated = np.vstack([nominated, screen[order]])
        perturb = np.clip(np.repeat(nominated, 2, axis=0)
                          + rng.normal(0, .04, (2 * len(nominated), n)), -B, B)
        data = np.vstack([data, nominated, perturb])
        theta, training = train_metric(theta, data, dynamics, alpha, rate, margin, steps, lr)
        row["training"] = training
        if iteration > 0 and training["final_empirical_loss"] == 0.0 and training["steps"] == 1:
            # The finite-step learner has made no update. Repeating the same
            # deterministic oracle cannot add evidence; stop with its explicit
            # inconclusive status rather than spending the remaining budget.
            stop_reason = "zero_loss_unchanged_parameter"
            break
    guarantees = (coarse_metric_guarantees(factor_metric(theta, alpha), B, alpha, rate,
                                          verification["uniform_raw_margin_lower"])
                  if verification["certified"] else None)
    return {"seed": seed, "certified": verification["certified"],
            "rounds": iteration, "theta": theta.tolist(), "trace": trace,
            "final_oracle": verification, "domain": [-B, B], "a": alpha,
            "rate": rate, "training_margin": margin, "guarantees": guarantees,
            "stop_reason": stop_reason,
            "protocol": {"test": "normalized", "margin_rule": "fixed", "margin": margin,
                         "initial_samples": initial_samples, "steps_per_round": steps,
                         "max_rounds": max_rounds, "learning_rate": lr,
                         "optimizer_state": "reset each round", "diagnostic_axis_points": 9,
                         "unresolved_point_cap": 60, "nominal_point_cap": 60,
                         "perturbations_per_point": 2, "perturbation_std": .04,
                         "oracle_budget": vars(budget)}}


def subdivided_scalar_bound(poly, B, depth=3):
    """Certified absolute bound using a fixed dyadic partition (all dimensions)."""
    n = poly.n
    axis = np.linspace(-B, B, 2 ** depth + 1)
    out = 0.0
    for cell in product(range(len(axis) - 1), repeat=n):
        lo = [axis[k] for k in cell]
        hi = [axis[k + 1] for k in cell]
        enclosure = poly.enclosure(lo, hi)
        out = max(out, float(enclosure.absmax()))
    return out


def metric_constants(metric, B, depth=3):
    n = len(metric)
    frob2 = sum((p * p for row in metric for p in row), ExactPoly.constant(n, 0))
    b = sqrt_up(subdivided_scalar_bound(frob2, B, depth))
    derivative2 = sum((p.derivative(k) ** 2 for k in range(n) for row in metric for p in row),
                      ExactPoly.constant(n, 0))
    mu1 = sqrt_up(subdivided_scalar_bound(derivative2, B, depth))
    return b, mu1


def coarse_metric_guarantees(metric, B, a, rate, eta):
    """Cheap global Frobenius bound, with directed rate and overshoot export."""
    squared = Iv(0.)
    for row in metric:
        for p in row:
            value = float(p.enclosure([-B] * len(metric), [B] * len(metric)).absmax())
            squared = squared + Iv(value).square()
    b = float(squared.sqrt().hi)
    rate_lower = float_down(rational(rate) + rational(eta) / rational(mul_up(2., b)))
    overshoot = sqrt_up(float_up(rational(b) / rational(a)))
    return {"b_upper": b, "eta_lower": eta, "a": a, "rate_lower": rate_lower,
            "overshoot_upper": overshoot,
            "bound_method": "global exact recentered entry bounds and outward Frobenius norm"}


def residual_constants(true_dynamics, fitted_dynamics, B, depth=3):
    n = len(true_dynamics)
    residual = [t - f for t, f in zip(true_dynamics, fitted_dynamics)]
    zero = ExactPoly.constant(n, 0)
    norm2 = sum((p * p for p in residual), zero)
    derivative2 = sum((p.derivative(k) ** 2 for p in residual for k in range(n)), zero)
    return {"delta0_upper": sqrt_up(subdivided_scalar_bound(norm2, B, depth)),
            "delta1_upper": sqrt_up(subdivided_scalar_bound(derivative2, B, depth)),
            "residual": [p.serializable() for p in residual],
            "jacobian_norm": "spectral norm bounded by Frobenius norm",
            "subdivision_depth": depth, "bound_scope": "global on the declared box",
            "requires_known_polynomial_truth": True}


def transfer_guarantee(eta, b, mu1, delta0, delta1, rate, a):
    value_part = mul_up(mu1, delta0)
    slope_part = mul_up(mul_up(2, b), delta1)
    consumed = add_up(value_part, slope_part)
    remainder = sub_down(eta, consumed)
    rate_lower = float_down(rational(rate) + rational(remainder) / rational(mul_up(2, b))) if remainder > 0 else None
    overshoot = sqrt_up(float_up(rational(b) / rational(a)))
    return {"certified": remainder > 0, "eta_lower": eta, "b_upper": b,
            "mu1_upper": mu1, "delta0_upper": delta0, "delta1_upper": delta1,
            "value_budget_upper": value_part, "slope_budget_upper": slope_part,
            "consumed_budget_upper": consumed, "remaining_margin_lower": remainder,
            "true_rate_lower": rate_lower, "overshoot_upper": overshoot,
            "finite_distance_scope": "requires a connecting family whose true-flow image stays inside the verification domain"}


def trajectory_samples(seed=0, trajectories=12, times=25, derivative_dt=.01):
    """Observe exact position trajectories and estimate derivatives centrally."""
    rng = np.random.RandomState(seed)
    starts = rng.uniform(-1.5, 1.5, (trajectories, 2))
    ts = np.linspace(.05, 1.5, times)
    def position(t):
        e = np.exp(-t)
        return np.stack([e * starts[:, 0],
                         e * (starts[:, 1] + starts[:, 0] ** 2) - e ** 2 * starts[:, 0] ** 2], axis=1)
    X, derivative = [], []
    for t in ts:
        X.append(position(t))
        derivative.append((position(t + derivative_dt) - position(t - derivative_dt)) / (2 * derivative_dt))
    return np.concatenate(X), np.concatenate(derivative), starts, ts


def fit_surrogate(seed=0, derivative_dt=.01):
    X, Y, starts, ts = trajectory_samples(seed, derivative_dt=derivative_dt)
    exponents = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    design = np.stack([X[:, 0] ** i * X[:, 1] ** j for i, j in exponents], axis=1)
    coefficients, _, rank, _ = np.linalg.lstsq(design, Y, rcond=None)
    if rank != len(exponents):
        raise ArithmeticError("rank deficient trajectory regression")
    dynamics = [ExactPoly(2, {k: coefficients[j, i] for j, k in enumerate(exponents)}) for i in range(2)]
    return dynamics, {"seed": seed, "method": "ordinary least squares on quadratic monomials",
                      "derivative_observations": "central differences of sampled exact trajectory positions",
                      "derivative_dt": derivative_dt, "position_noise": 0.0,
                      "trajectory_count": len(starts), "times_per_trajectory": len(ts),
                      "sample_count": len(X), "design_rank": int(rank),
                      "training_rmse": float(np.sqrt(np.mean((design @ coefficients - Y) ** 2))),
                      "coefficients": coefficients.T.tolist(), "feature_exponents": exponents,
                      "initial_states": starts.tolist(), "sample_times": ts.tolist(),
                      "learned_model": [p.serializable() for p in dynamics],
                      "truth_not_used_in_regression": True}


def run_surrogate(seed=0, budget=OracleBudget(target_margin=.005)):
    fitted, fitting = fit_surrogate(seed)
    learned = run_learned_metric(fitted, seed, B=2., margin=.05, max_rounds=8,
                                steps=400, initial_samples=400, budget=budget)
    metric = factor_metric(np.asarray(learned["theta"]), .1)
    residual = residual_constants(benchmark_dynamics(2), fitted, 2., depth=3)
    transfer = None
    if learned["certified"]:
        b, mu1 = metric_constants(metric, 2., depth=3)
        transfer = transfer_guarantee(learned["final_oracle"]["uniform_raw_margin_lower"], b, mu1,
                                      residual["delta0_upper"], residual["delta1_upper"], .4, .1)
    return {"fitting": fitting, "metric_learning": learned, "residual_bounds": residual,
            "transfer": transfer, "interpretation": "Known-truth benchmark: fitted surrogate is learned from trajectory observations; global residual/Jacobian certification uses the independently known polynomial truth."}


def compare_frozen_2d_metrics(max_metrics=3, budget=OracleBudget(target_margin=.001)):
    import cegis_c
    path = ROOT / "results/traces.json"
    original = path.read_bytes()
    traces = json.loads(original)
    selected = [(int(seed), run) for seed, run in traces["regimes"]["certify"].items() if run["certified"]]
    selected.sort()
    records = []
    for seed, run in selected[:max_metrics]:
        theta = np.asarray(run["theta"])
        contraction = polynomial_matrix_contraction(factor_metric(theta, .1), benchmark_dynamics(2), .4)
        started = time.perf_counter()
        grid = cegis_c.verify(theta, 2001, normalized=True, certified=True, cert_depth=3)
        grid_s = time.perf_counter() - started
        adaptive = adaptive_oracle(contraction, [-2, -2], [2, 2], budget,
                                    check_points=diagnostic_grid(2, [-2, -2], [2, 2]))
        records.append({"seed": seed, "theta": theta.tolist(),
                        "grid": {"certified": bool(grid["certified"]), "axis_points": 2001,
                                 "nominal_grid_evaluations": 2001 ** 2,
                                 "outward_grid_evaluations": 2001 ** 2 if grid["grid_enclosed"] else 0,
                                 "wall_s": grid_s, "Lh_upper": grid["L"], "tau_upper": grid["tau"],
                                 "normalized_slack_lower": grid["uniform_slack"]},
                        "adaptive": adaptive})
    if len(records) != max_metrics:
        raise ValueError("insufficient certified frozen metrics")
    return {"input": "results/traces.json", "input_sha256": hashlib.sha256(original).hexdigest(),
            "selection": "first certified seeds in ascending order", "records": records,
            "comparison_scope": "identical frozen metrics and domain; grid proves base-rate inequality with a normalized Lipschitz margin, adaptive proves a prescribed positive raw margin; work counts have different units",
            "worst_case": "adaptive box subdivision remains exponential; these timings do not establish dimension-independent complexity"}


def replay_oracle_certificate(result, contraction):
    """Independently replay every positive box decision and its complete cover."""
    if result.get("status") not in ("certified", "inconclusive"):
        raise ValueError("invalid adaptive status")
    if result.get("certified") != (result["status"] == "certified") or result.get("outward_rounded") is not True:
        raise ValueError("unsafe adaptive status")
    proof = result.get("proof", {})
    if proof.get("polynomial_sha256") != polynomial_matrix_hash(contraction):
        raise ValueError("adaptive certificate polynomial mismatch")
    budget = result.get("budget", {})
    margin = result.get("target_margin")
    counters = result.get("counters", {})
    if not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
        raise ValueError("invalid adaptive margin")
    if (not isinstance(budget.get("max_boxes"), int) or budget["max_boxes"] < 1
            or not isinstance(budget.get("max_depth"), int) or budget["max_depth"] < 0
            or any(not isinstance(counters.get(k), int) or counters[k] < 0
                   for k in ("boxes_evaluated", "accepted_leaves", "splits", "maximum_depth_reached", "nominal_point_evaluations"))
            or counters["maximum_depth_reached"] > budget["max_depth"]):
        raise ValueError("invalid adaptive work accounting")
    if budget.get("target_margin") != margin or counters.get("boxes_evaluated", -1) > budget.get("max_boxes", -2):
        raise ValueError("inconsistent adaptive budget")
    if not result["certified"]:
        if result.get("uniform_raw_margin_lower") is not None or proof.get("leaf_paths") != []:
            raise ValueError("inconclusive result claims a positive bound")
        if not result.get("training_points"):
            raise ValueError("inconclusive result lacks diagnostic points")
        return 0
    if result.get("reason") != "all_leaves_certified" or result.get("uniform_raw_margin_lower") != margin:
        raise ValueError("invalid positive adaptive margin")
    paths = proof.get("leaf_paths")
    if not isinstance(paths, list) or not paths or len(paths) != len(set(paths)):
        raise ValueError("missing or duplicate partition leaves")
    if any(not isinstance(p, str) or any(c not in "01" for c in p)
           or len(p) > budget["max_depth"] for p in paths):
        raise ValueError("invalid partition path")
    ordered = sorted(paths)
    if any(q.startswith(p) for p, q in zip(ordered, ordered[1:])):
        raise ValueError("overlapping partition leaves")
    if sum((Fraction(1, 2 ** len(p)) for p in paths), Fraction(0)) != 1:
        raise ValueError("incomplete partition cover")
    leaves = len(paths)
    if counters.get("accepted_leaves") != leaves or counters.get("splits") != leaves - 1 or counters.get("boxes_evaluated") != 2 * leaves - 1:
        raise ValueError("partition work counts inconsistent")
    lo0 = np.asarray(result["domain_lower"], float)
    hi0 = np.asarray(result["domain_upper"], float)
    n = len(contraction)
    if lo0.shape != (n,) or hi0.shape != (n,) or np.any(lo0 >= hi0) or not np.isfinite([lo0, hi0]).all():
        raise ValueError("invalid replay domain")
    weights = _split_weights(contraction, lo0, hi0)
    if proof.get("split_weights") != weights.tolist():
        raise ValueError("subdivision heuristic mismatch")
    leaf_set = set(paths)
    stack = [(lo0, hi0, "", _initial_bernstein_matrix(contraction, lo0, hi0))]
    while stack:
        lo, hi, path, coefficients = stack.pop()
        if path in leaf_set:
            if not negative_definite_interval(_bernstein_matrix_enclosure(coefficients), margin):
                raise ValueError("positive adaptive leaf fails independent replay")
            continue
        k = int(np.argmax((hi - lo) * weights))
        midpoint = lo[k] + (hi[k] - lo[k]) / 2
        if not lo[k] < midpoint < hi[k]:
            raise ValueError("non-splitting certificate path")
        ratio = (rational(midpoint) - rational(lo[k])) / (rational(hi[k]) - rational(lo[k]))
        left_coeff, right_coeff = _split_bernstein_matrix(coefficients, k, ratio)
        left_hi, right_lo = hi.copy(), lo.copy()
        left_hi[k] = midpoint; right_lo[k] = midpoint
        stack.append((right_lo, hi.copy(), path + "1", right_coeff))
        stack.append((lo.copy(), left_hi, path + "0", left_coeff))
    return leaves


def _polynomial_from_serialized(n, records):
    coeffs = {}
    for row in records:
        key = tuple(row["powers"])
        if key in coeffs:
            raise ValueError("duplicate serialized monomial")
        coeffs[key] = Fraction(int(row["numerator"]), int(row["denominator"]))
    return ExactPoly(n, coeffs)


def _check_learned_run_certificate(run, dynamics, B, expected_shape):
    theta = np.asarray(run["theta"], float)
    if theta.shape != expected_shape or not np.isfinite(theta).all():
        raise ValueError("invalid learned metric")
    if run["domain"] != [-B, B] or run["a"] != .1 or run["rate"] != .4:
        raise ValueError("metric scope mismatch")
    oracle = run["final_oracle"]
    if run["certified"] != oracle["certified"] or run["trace"][-1]["oracle"] != oracle:
        raise ValueError("run certification mismatch")
    if len(run["trace"]) != run["rounds"] + 1 or [t["round"] for t in run["trace"]] != list(range(run["rounds"] + 1)):
        raise ValueError("round accounting mismatch")
    n = len(dynamics)
    if oracle["domain_lower"] != [-B] * n or oracle["domain_upper"] != [B] * n:
        raise ValueError("adaptive domain mismatch")
    metric = factor_metric(theta, .1)
    contraction = polynomial_matrix_contraction(metric, dynamics, .4)
    expected = (coarse_metric_guarantees(metric, B, .1, .4, oracle["uniform_raw_margin_lower"])
                if oracle["certified"] else None)
    if run.get("guarantees") != expected:
        raise ValueError("global metric/rate guarantee mismatch")
    return replay_oracle_certificate(oracle, contraction)


def _check_3d_run_certificate(run):
    return _check_learned_run_certificate(run, benchmark_dynamics(3), 1., (6, 4))


def validate_results(data, replay_workers=1):
    """Fail closed on missing/unsafe evidence and replay every positive leaf.

    This validation is deliberately more than a schema check. Exact contraction
    matrices are rebuilt from the stored real parameters, and each complete
    partition certificate is checked again with outward interval arithmetic.
    """
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite extension result")
        if isinstance(value, dict):
            for v in value.values(): finite(v)
        elif isinstance(value, list):
            for v in value: finite(v)
    finite(data)
    if data.get("schema_version") != 1 or data.get("verification_backend", {}).get("outward_rounded") is not True:
        raise ValueError("missing directed extension backend")
    cfg = data["config"]
    if data["three_dimensional"]["reference"] != independent_pullback_identity():
        raise ValueError("independent known-metric identity mismatch")
    runs = data["three_dimensional"]["runs"]
    if len(runs) != cfg["n_seeds"] or [r["seed"] for r in runs] != cfg["seeds"] or cfg["seeds"] != list(range(cfg["n_seeds"])):
        raise ValueError("3D seed accounting mismatch")
    if cfg["domain_3d"] != [-1., 1.] or cfg["alpha"] != .1 or cfg["rate"] != .4:
        raise ValueError("undeclared benchmark domain or constants")
    replayed = 0
    if not isinstance(replay_workers, int) or replay_workers < 1:
        raise ValueError("positive validation worker count required")
    if replay_workers == 1 or len(runs) == 1:
        replayed += sum(_check_3d_run_certificate(run) for run in runs)
    else:
        with ProcessPoolExecutor(max_workers=min(replay_workers, len(runs)), initializer=self_test) as pool:
            replayed += sum(pool.map(_check_3d_run_certificate, runs))
    count = sum(r["certified"] for r in runs)
    summary = data["three_dimensional"]
    if count != summary["certified_count"] or len(runs) - count != summary["uncertified_count"] or summary["certified_rounds"] != [r["rounds"] for r in runs if r["certified"]]:
        raise ValueError("3D outcome summary mismatch")
    if count < 1:
        raise ValueError("no certified 3D output for the extension study")
    comparison = data["frozen_oracle_comparison"]
    path = ROOT / comparison["input"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != comparison["input_sha256"]:
        raise ValueError("frozen baseline source mismatch")
    source = json.loads(path.read_text())
    if not comparison["records"]:
        raise ValueError("missing matched oracle comparison")
    for row in comparison["records"]:
        saved = source["regimes"]["certify"][str(row["seed"])]
        if not saved["certified"] or row["theta"] != saved["theta"]:
            raise ValueError("frozen metric selection mismatch")
        grid = row["grid"]
        if not grid["certified"] or grid["axis_points"] != 2001 or grid["outward_grid_evaluations"] != 2001 ** 2 or grid["normalized_slack_lower"] < 0:
            raise ValueError("unsafe reference grid certificate")
        C = polynomial_matrix_contraction(factor_metric(np.asarray(row["theta"]), .1), benchmark_dynamics(2), .4)
        replayed += replay_oracle_certificate(row["adaptive"], C)
    surrogate = data["learned_surrogate"]
    fitting = surrogate["fitting"]
    fitted = [_polynomial_from_serialized(2, p) for p in fitting["learned_model"]]
    coefficients = [[p.coeffs.get(tuple(k), Fraction(0)) for k in fitting["feature_exponents"]] for p in fitted]
    if [[float(c) for c in row] for row in coefficients] != fitting["coefficients"] or fitting.get("truth_not_used_in_regression") is not True:
        raise ValueError("fitted model evidence mismatch")
    if fitting["sample_count"] != fitting["trajectory_count"] * fitting["times_per_trajectory"] or fitting["design_rank"] != len(fitting["feature_exponents"]):
        raise ValueError("trajectory fitting configuration mismatch")
    residual = residual_constants(benchmark_dynamics(2), fitted, 2., depth=3)
    if residual != surrogate["residual_bounds"]:
        raise ValueError("global residual bound mismatch")
    learned = surrogate["metric_learning"]
    replayed += _check_learned_run_certificate(learned, fitted, 2., (3, 3))
    if not learned["certified"] or surrogate.get("transfer") is None:
        raise ValueError("learned model has no certified transfer output")
    b, mu1 = metric_constants(factor_metric(np.asarray(learned["theta"]), .1), 2., depth=3)
    transfer = transfer_guarantee(learned["final_oracle"]["uniform_raw_margin_lower"], b, mu1,
                                  residual["delta0_upper"], residual["delta1_upper"], .4, .1)
    if transfer != surrogate["transfer"] or not transfer["certified"]:
        raise ValueError("residual transfer inequality not certified")
    return {"three_dimensional_certified": count, "three_dimensional_total": len(runs),
            "matched_metrics": len(comparison["records"]), "replayed_certified_leaves": replayed,
            "surrogate_transfer_certified": True}


def make_figures(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = ROOT / "paper"
    out.mkdir(exist_ok=True)
    certified = [r for r in report["three_dimensional"]["runs"] if r["certified"]]
    if certified:
        selected = certified[0]
        axis = np.linspace(-1, 1, 101)
        x, y = np.meshgrid(axis, axis)
        points = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
        metric = factor_metric(np.asarray(selected["theta"]), .1)
        contraction = polynomial_matrix_contraction(metric, benchmark_dynamics(3), .4)
        values = np.linalg.eigvalsh(matrix_values(contraction, points))[:, -1].reshape(x.shape)
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        contour = ax.contourf(x, y, values, levels=16)
        fig.colorbar(contour, ax=ax, label=r"$\lambda_{\max}(C_\theta(x_1,x_2,0))$")
        ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$", title="Certified 3D metric: diagnostic slice at $x_3=0$")
        fig.tight_layout()
        fig.savefig(out / "fig_extension3d.pdf")
        plt.close(fig)
    surrogate = report["learned_surrogate"]
    transfer = surrogate["transfer"]
    if transfer is not None:
        fig, ax = plt.subplots(figsize=(5.4, 3.5))
        labels = [r"$\eta$", r"$\mu_1\delta_0$", r"$2b\delta_1$", "remaining"]
        values = [transfer["eta_lower"], transfer["value_budget_upper"],
                  transfer["slope_budget_upper"], transfer["remaining_margin_lower"]]
        ax.bar(labels, values, color=["#3567a0", "#ca843a", "#ca843a", "#448b61"])
        ax.set(ylabel="Certified raw margin / error budget",
               title="Trajectory-fitted model:\ncertified transfer to known true dynamics")
        fig.tight_layout()
        fig.savefig(out / "fig_surrogate.pdf")
        plt.close(fig)


def _run_3d_seed(seed):
    run = run_learned_metric(benchmark_dynamics(3), seed, B=1., budget=OracleBudget())
    print(f"3D seed={seed} certified={run['certified']} rounds={run['rounds']} boxes={run['final_oracle']['counters']['boxes_evaluated']}", flush=True)
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="one 3D seed and one frozen comparison")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4, help="independent 3D seed processes; matched 2D timings remain serial")
    parser.add_argument("--validate-only", action="store_true", help="replay existing complete certificates without training")
    args = parser.parse_args(argv)
    if args.seeds < 1 or args.workers < 1:
        parser.error("at least one seed and worker required")
    if args.validate_only:
        data = json.loads((ROOT / "results/revision_extensions.json").read_text())
        print(json.dumps(validate_results(data, replay_workers=args.workers), sort_keys=True), flush=True)
        return data
    started = time.perf_counter()
    self_test()
    n_seeds = 1 if args.pilot else args.seeds
    budget = OracleBudget()
    config = {"n_seeds": n_seeds, "seeds": list(range(n_seeds)), "domain_3d": [-1., 1.],
              "alpha": .1, "rate": .4, "training_margin_3d": .005,
              "initial_samples_3d": 256, "steps_per_round_3d": 300,
              "max_retraining_rounds_3d": 8, "learning_rate": .02,
              "initialization_std": .01, "screen_axis_points": 9,
              "returned_point_cap": 60, "diagnostic_worst_point_cap": 60,
              "perturbations_per_nominated_point": 2, "perturbation_std": .04,
              "initialization": "identity lower factor plus seeded N(0,0.01^2) coefficient perturbations",
              "metric_parameters_3d": 24, "adaptive_budget": vars(budget),
              "surrogate_seed": 0, "surrogate_derivative_dt": .01,
              "three_dimensional_worker_processes": min(args.workers, n_seeds),
              "learner_early_stop": "stop after an inconclusive oracle if a zero-loss retraining leaves parameters unchanged",
              "pilot": args.pilot}
    print("EXTENSION_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
    reference = independent_pullback_identity()
    if min(args.workers, n_seeds) == 1:
        runs = [_run_3d_seed(seed) for seed in range(n_seeds)]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, n_seeds), initializer=self_test) as pool:
            runs = list(pool.map(_run_3d_seed, range(n_seeds)))
    comparisons = compare_frozen_2d_metrics(1 if args.pilot else 3)
    print("FROZEN_COMPARISON complete", flush=True)
    surrogate = run_surrogate()
    print("SURROGATE certified=" + str(surrogate["metric_learning"]["certified"])
          + " transfer=" + str(surrogate["transfer"] is not None and surrogate["transfer"]["certified"]), flush=True)
    report = {"schema_version": 1, "config": config, "verification_backend": backend_info(),
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "numpy": np.__version__},
              "three_dimensional": {"reference": reference, "runs": runs,
                                    "certified_count": sum(r["certified"] for r in runs),
                                    "uncertified_count": sum(not r["certified"] for r in runs),
                                    "certified_rounds": [r["rounds"] for r in runs if r["certified"]],
                                    "finite_distance_scope": "variational contraction on the box; finite distances require the connecting-flow condition"},
              "frozen_oracle_comparison": comparisons, "learned_surrogate": surrogate,
              "wall_s": time.perf_counter() - started}
    output = ROOT / "results/revision_extensions.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    make_figures(report)
    print("WROTE " + str(output), flush=True)
    return report


if __name__ == "__main__":
    main()
