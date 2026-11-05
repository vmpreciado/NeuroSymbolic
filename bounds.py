"""bounds.py: certified polynomial bounds for the CEGIS-C verifier.

Implements
  - a nominal polynomial class Poly2 for diagnostics and ExactPoly2 for
    coefficient assembly, differentiation, and binomial shifts in exact
    rational arithmetic;
  - outward conversion of coefficient/subdivision bounds to binary64;
  - symbolic assembly of the contraction matrix C(x) and its partial
    derivatives from a numeric theta;
  - certified Lipschitz upper bounds for the raw test g = lambda_max(C)
    and the normalized test gt = g / tr M.

Certificate functions use ExactPoly2 from the original stored binary64 theta,
not widened coefficients of an already rounded polynomial. Grid evaluation
uses outward coefficient enclosures and interval Horner arithmetic. Scalar
norms, quotients, and square roots round in the sufficient direction.
All bounds cover x in [-BOX, BOX]^2; default subdivision depth is three.

Soundness sketch (proofs in the paper).

Raw bound.
  C(x) is symmetric 2x2 with entries (c11, c12, c22) polynomial in x.
  |g(x) - g(y)| <= ||C(x) - C(y)||_2 <= ||C(x) - C(y)||_F.  By
  Cauchy-Schwarz over coordinates,
      |g(x) - g(y)| <= sqrt( sum_k sup_x ||dC/dx_k||_F^2 ) * ||x-y||,
  and sup_x ||dC/dx_k||_F^2 <= D_k^2 := supabs(dc11/dxk)^2
  + 2 supabs(dc12/dxk)^2 + supabs(dc22/dxk)^2, since |a|^2 + 2|b|^2 + |c|^2
  is sublinear in entrywise supabs.

Normalized bound.
  gt = g/v with v = tr M.  Quotient inequality with v_min := n*ALPHA = 0.2
  (exact lower bound from M = ALPHA*I + L^T L) and u_max := sup|g| <=
  sup ||C||_F, plus Lv := Lipschitz of v, gives
      Lgt <= Lg / v_min + u_max * Lv / v_min^2.
"""
import math
from fractions import Fraction
import numpy as np

from rounding import (Interval, float_down, float_up, sqrt_up, div_up,
                      add_up, mul_up, sub_down)

ALPHA = 0.1
RATE = 0.4
BOX = 2.0


# ----------------------- polynomial class -----------------------

class Poly2:
    """Polynomial in (x1, x2) as dict[(i, j)] -> float, meaning
    sum coeff * x1^i * x2^j. Zero coefficients are pruned."""

    __slots__ = ("coeffs",)

    def __init__(self, coeffs=None):
        self.coeffs = {}
        if coeffs:
            for k, v in coeffs.items():
                if v != 0.0:
                    self.coeffs[k] = float(v)

    # --- constructors ---
    @classmethod
    def zero(cls):
        return cls()

    @classmethod
    def const(cls, c):
        if c == 0.0:
            return cls()
        return cls({(0, 0): float(c)})

    @classmethod
    def x1(cls):
        return cls({(1, 0): 1.0})

    @classmethod
    def x2(cls):
        return cls({(0, 1): 1.0})

    # --- internals ---
    def _addto(self, k, v):
        new = self.coeffs.get(k, 0.0) + v
        if new == 0.0:
            self.coeffs.pop(k, None)
        else:
            self.coeffs[k] = new

    # --- arithmetic ---
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Poly2.const(other)
        out = Poly2(self.coeffs)
        for k, v in other.coeffs.items():
            out._addto(k, v)
        return out

    __radd__ = __add__

    def __neg__(self):
        return Poly2({k: -v for k, v in self.coeffs.items()})

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self + (-float(other))
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            if other == 0.0:
                return Poly2()
            return Poly2({k: v * other for k, v in self.coeffs.items()})
        out = Poly2()
        for k1, v1 in self.coeffs.items():
            for k2, v2 in other.coeffs.items():
                out._addto((k1[0] + k2[0], k1[1] + k2[1]), v1 * v2)
        return out

    __rmul__ = __mul__

    # --- calculus and evaluation ---
    def deriv(self, var):
        """Partial derivative w.r.t. x1 (var=0) or x2 (var=1)."""
        out = Poly2()
        if var == 0:
            for (i, j), v in self.coeffs.items():
                if i >= 1:
                    out.coeffs[(i - 1, j)] = i * v
        else:
            for (i, j), v in self.coeffs.items():
                if j >= 1:
                    out.coeffs[(i, j - 1)] = j * v
        return out

    def eval(self, x1, x2):
        s = 0.0
        for (i, j), v in self.coeffs.items():
            s += v * (x1 ** i) * (x2 ** j)
        return s

    def evalv(self, X):
        """Vectorized evaluation on X shape (N, 2)."""
        x1, x2 = X[:, 0], X[:, 1]
        out = np.zeros(X.shape[0])
        for (i, j), v in self.coeffs.items():
            out = out + v * (x1 ** i) * (x2 ** j)
        return out

    # --- bounds ---
    def supabs_box(self, B):
        """Coefficient upper bound on sup_{|x1|,|x2|<=B} |p(x)|."""
        s = 0.0
        for (i, j), v in self.coeffs.items():
            s += abs(v) * (B ** (i + j))
        return s

    def shift(self, c1, c2):
        """Return polynomial q(y1, y2) := p(c1 + y1, c2 + y2)."""
        out = Poly2()
        for (i, j), v in self.coeffs.items():
            # binomial expansion of (c1+y1)^i (c2+y2)^j
            for a in range(i + 1):
                ca = math.comb(i, a) * (c1 ** (i - a))
                if ca == 0.0:
                    continue
                for b in range(j + 1):
                    cb = math.comb(j, b) * (c2 ** (j - b))
                    if cb == 0.0:
                        continue
                    out._addto((a, b), v * ca * cb)
        return out

    def supabs_subdiv(self, B, depth):
        """Sound upper bound on sup_{[-B,B]^2} |p|, by splitting the box
        into (2^depth) x (2^depth) equal subboxes, recentering on each,
        and applying supabs_box at the subbox half-width."""
        if depth <= 0:
            return self.supabs_box(B)
        n = 2 ** depth
        h = B / n
        best = 0.0
        for i in range(n):
            c1 = -B + (2 * i + 1) * h
            for j in range(n):
                c2 = -B + (2 * j + 1) * h
                b = self.shift(c1, c2).supabs_box(h)
                if b > best:
                    best = b
        return best

    def inf_subdiv(self, B, depth):
        """Sound lower bound on inf_{[-B,B]^2} p(x) via subdivision: on
        each subbox of half-width h centered at c, use
            p(x) >= p(c) - supabs(p - p(c))
        evaluated at half-width h. Take the min over subboxes."""
        def _on_box(poly, B_local):
            const = poly.coeffs.get((0, 0), 0.0)
            rest = Poly2({k: v for k, v in poly.coeffs.items() if k != (0, 0)})
            return const - rest.supabs_box(B_local)

        if depth <= 0:
            return _on_box(self, B)
        n = 2 ** depth
        h = B / n
        best = None
        for i in range(n):
            c1 = -B + (2 * i + 1) * h
            for j in range(n):
                c2 = -B + (2 * j + 1) * h
                lb = _on_box(self.shift(c1, c2), h)
                if best is None or lb < best:
                    best = lb
        return best


# ----------------------- exact coefficient arithmetic -----------------------


def _fraction(value):
    """Stored binary64 inputs denote their exact real (dyadic) values."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite polynomial coefficient")
    return Fraction.from_float(value)


class ExactPoly2:
    """Exact rational coefficients, including all assembly and binomial shifts.

    Bound methods round once outward when converting an exact rational bound
    to binary64. ``evalv`` is diagnostic only; interval_eval certifies values.
    """
    __slots__ = ("coeffs",)

    def __init__(self, coeffs=None):
        self.coeffs = {k: _fraction(v) for k, v in (coeffs or {}).items() if v != 0}

    @classmethod
    def zero(cls):
        return cls()

    @classmethod
    def const(cls, value):
        return cls({(0, 0): value})

    @classmethod
    def x1(cls):
        return cls({(1, 0): 1})

    @classmethod
    def x2(cls):
        return cls({(0, 1): 1})

    def _addto(self, key, value):
        value = self.coeffs.get(key, Fraction(0)) + value
        if value:
            self.coeffs[key] = value
        else:
            self.coeffs.pop(key, None)

    def __add__(self, other):
        if not isinstance(other, ExactPoly2):
            other = ExactPoly2.const(other)
        out = ExactPoly2(self.coeffs)
        for key, value in other.coeffs.items():
            out._addto(key, value)
        return out

    __radd__ = __add__

    def __neg__(self):
        return ExactPoly2({key: -value for key, value in self.coeffs.items()})

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if not isinstance(other, ExactPoly2):
            value = _fraction(other)
            return ExactPoly2({key: coeff * value for key, coeff in self.coeffs.items()})
        out = ExactPoly2()
        for (i, j), value in self.coeffs.items():
            for (a, b), coeff in other.coeffs.items():
                out._addto((i + a, j + b), value * coeff)
        return out

    __rmul__ = __mul__

    def deriv(self, var):
        if var not in (0, 1):
            raise ValueError("derivative coordinate must be 0 or 1")
        out = ExactPoly2()
        for (i, j), value in self.coeffs.items():
            degree = (i, j)[var]
            if degree:
                key = (i - 1, j) if var == 0 else (i, j - 1)
                out._addto(key, degree * value)
        return out

    def eval(self, x1, x2):
        x1, x2 = _fraction(x1), _fraction(x2)
        return sum((v * x1**i * x2**j for (i, j), v in self.coeffs.items()), Fraction(0))

    def evalv(self, X):
        """Nearest-rounded diagnostic evaluation, never used to certify."""
        out = np.zeros(len(X))
        for (i, j), value in self.coeffs.items():
            out += float(value) * X[:, 0]**i * X[:, 1]**j
        return out

    def interval_eval(self, X):
        """Nested Horner evaluation with outward coefficient conversions."""
        if not self.coeffs:
            return Interval(np.zeros(len(X)))
        x, y = Interval(X[:, 0]), Interval(X[:, 1])
        by_x = {}
        for (i, j), value in self.coeffs.items():
            by_x.setdefault(i, {})[j] = value
        def coefficient(value):
            return Interval(float_down(value), float_up(value))
        def row(i):
            coeffs = by_x.get(i, {})
            if not coeffs:
                return Interval(0.0)
            degree = max(coeffs)
            result = coefficient(coeffs[degree])
            for j in range(degree - 1, -1, -1):
                result = result * y
                if j in coeffs:
                    result = result + coefficient(coeffs[j])
            return result
        degree = max(by_x)
        result = row(degree)
        for i in range(degree - 1, -1, -1):
            result = result * x
            if i in by_x:
                result = result + row(i)
        # A constant polynomial still has a scalar interval; broadcast it.
        return Interval(np.broadcast_to(result.lo, (len(X),)),
                        np.broadcast_to(result.hi, (len(X),)))

    def shift(self, c1, c2):
        c1, c2 = _fraction(c1), _fraction(c2)
        out = ExactPoly2()
        for (i, j), value in self.coeffs.items():
            for a in range(i + 1):
                ca = math.comb(i, a) * c1**(i - a)
                for b in range(j + 1):
                    out._addto((a, b), value * ca * math.comb(j, b) * c2**(j - b))
        return out

    def supabs_fraction(self, B):
        B = _fraction(B)
        if B < 0:
            raise ValueError("negative box half-width")
        return sum((abs(v) * B**(i + j) for (i, j), v in self.coeffs.items()), Fraction(0))

    def supabs_box(self, B):
        return float_up(self.supabs_fraction(B))

    def inf_fraction(self, B):
        return self.coeffs.get((0, 0), Fraction(0)) - sum(
            (abs(v) * _fraction(B)**(i + j) for (i, j), v in self.coeffs.items()
             if (i, j) != (0, 0)), Fraction(0))

    def inf_box(self, B):
        return float_down(self.inf_fraction(B))

    def supabs_subdiv(self, B, depth):
        centers, h = _exact_subboxes(B, depth)
        return float_up(max(self.shift(c1, c2).supabs_fraction(h) for c1, c2 in centers))

    def inf_subdiv(self, B, depth):
        centers, h = _exact_subboxes(B, depth)
        return float_down(min(self.shift(c1, c2).inf_fraction(h) for c1, c2 in centers))


def _exact_subboxes(B, depth):
    if not isinstance(depth, int) or depth < 0:
        raise ValueError("subdivision depth must be a nonnegative integer")
    B = _fraction(B)
    if B <= 0:
        raise ValueError("box half-width must be positive")
    n = 2**depth
    h = B / n
    return [(-B + (2*i + 1)*h, -B + (2*j + 1)*h)
            for i in range(n) for j in range(n)], h


# ----------------------- symbolic assembly -----------------------

def _Lij(theta, poly=Poly2):
    """L_11, L_21, L_22 as poly. theta rows = coefficients on (1, x1, x2)."""
    feats = [poly.const(1.0), poly.x1(), poly.x2()]
    out = []
    for r in range(3):
        s = poly.zero()
        for k in range(3):
            if theta[r, k] != 0.0:
                s = s + feats[k] * float(theta[r, k])
        out.append(s)
    return out[0], out[1], out[2]


def assemble(theta, poly=Poly2):
    """Build all polynomial objects used by the bounds, given numeric theta.

    Returns a dict with keys
        m11, m12, m22       entries of M
        c11, c12, c22       entries of C
        dc                  list of (dc11/dxk, dc12/dxk, dc22/dxk), k=0,1
        v                   tr M
        dv                  [dv/dx1, dv/dx2]
        dm                  list of (dm11/dxk, dm12/dxk, dm22/dxk), k=0,1
    """
    L11, L21, L22 = _Lij(theta, poly=poly)
    m11 = poly.const(ALPHA) + L11 * L11 + L21 * L21
    m12 = L21 * L22
    m22 = poly.const(ALPHA) + L22 * L22
    dm = [(m11.deriv(k), m12.deriv(k), m22.deriv(k)) for k in range(2)]
    # f(x) = (-x1, -(x2 - x1^2))
    f1 = -poly.x1()
    f2 = -(poly.x2() - poly.x1() * poly.x1())
    f = [f1, f2]
    mdot11 = dm[0][0] * f[0] + dm[1][0] * f[1]
    mdot12 = dm[0][1] * f[0] + dm[1][1] * f[1]
    mdot22 = dm[0][2] * f[0] + dm[1][2] * f[1]
    # A(x) = [[-1, 0], [2 x1, -1]]
    a11 = poly.const(-1.0)
    a12 = poly.zero()
    a21 = poly.x1() * 2.0
    a22 = poly.const(-1.0)
    # S = A^T M + M A, symmetric 2x2
    s11 = a11 * m11 * 2.0 + a21 * m12 * 2.0
    s22 = a22 * m22 * 2.0 + a12 * m12 * 2.0
    s12 = a11 * m12 + a21 * m22 + a12 * m11 + a22 * m12
    c11 = mdot11 + s11 + m11 * (2.0 * RATE)
    c22 = mdot22 + s22 + m22 * (2.0 * RATE)
    c12 = mdot12 + s12 + m12 * (2.0 * RATE)
    dc = [(c11.deriv(k), c12.deriv(k), c22.deriv(k)) for k in range(2)]
    v = m11 + m22
    dv = [v.deriv(0), v.deriv(1)]
    return {
        "m11": m11, "m12": m12, "m22": m22,
        "c11": c11, "c12": c12, "c22": c22,
        "dc": dc, "v": v, "dv": dv, "dm": dm,
    }


def assemble_exact(theta):
    """Exact real polynomial for the stored binary64 candidate parameters."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (3, 3) or not np.isfinite(theta).all():
        raise ValueError("theta must be a finite 3x3 array")
    return assemble(theta, poly=ExactPoly2)


def _exact_model(theta, P):
    if P is not None and isinstance(P.get("m11"), ExactPoly2):
        return P
    return assemble_exact(theta)


def _norm_upper(values, weights=None):
    result = Interval(0.0)
    for value, weight in zip(values, weights or [1] * len(values)):
        result = result + Interval(value).square() * weight
    return float(result.sqrt().hi)


# ----------------------- certified constants -----------------------

def supabs(p, depth=3, B=BOX):
    return p.supabs_subdiv(B, depth)


def L_raw(theta, depth=3, B=BOX, P=None):
    """Outward Frobenius derivative bound after exact coefficient assembly."""
    P = _exact_model(theta, P)
    values, weights = [], []
    for row in P["dc"]:
        values.extend(supabs(entry, depth, B) for entry in row)
        weights.extend([1, 2, 1])
    return _norm_upper(values, weights)


def v_min_bound(theta, depth=3, B=BOX, P=None):
    P = _exact_model(theta, P)
    # Multiplication by two is exact for the fixed normal binary64 ALPHA.
    return max(2.0 * ALPHA, P["v"].inf_subdiv(B, depth))


def _subbox_centers(B, depth):
    """Exact rational centers and half-width, including non-dyadic B inputs."""
    return _exact_subboxes(B, depth)


def sup_abs_gt_bound(theta, depth=3, B=BOX, P=None):
    P = _exact_model(theta, P)
    pF2 = P["c11"] * P["c11"] + 2 * P["c12"] * P["c12"] + P["c22"] * P["c22"]
    centers, h = _exact_subboxes(B, depth)
    best = 0.0
    for c1, c2 in centers:
        sup_g2 = pF2.shift(c1, c2).supabs_box(h)
        inf_v = max(2.0 * ALPHA, P["v"].shift(c1, c2).inf_box(h))
        best = max(best, div_up(sqrt_up(sup_g2), inf_v))
    return best


def L_norm(theta, depth=3, B=BOX, P=None):
    P = _exact_model(theta, P)
    v_min = v_min_bound(theta, depth, B, P=P)
    Lg = L_raw(theta, depth, B, P=P)
    Lv = _norm_upper([supabs(entry, depth, B) for entry in P["dv"]])
    gt_sup = sup_abs_gt_bound(theta, depth, B, P=P)
    return div_up(add_up(Lg, mul_up(gt_sup, Lv)), v_min)


def lipschitz_certified(theta, depth=3, normalized=True, B=BOX):
    """Outward-rounded Lipschitz upper bound for the exact stored candidate."""
    return (L_norm if normalized else L_raw)(theta, depth=depth, B=B)


# ----------------------- additional certified ingredients -----------------------

def b_upper_bound(theta, depth=3, B=BOX, P=None):
    P = _exact_model(theta, P)
    return _norm_upper([supabs(P[k], depth, B) for k in ("m11", "m12", "m22")], [1, 2, 1])


def mu1_upper_bound(theta, depth=3, B=BOX, P=None):
    P = _exact_model(theta, P)
    values, weights = [], []
    for row in P["dm"]:
        values.extend(supabs(entry, depth, B) for entry in row)
        weights.extend([1, 2, 1])
    return _norm_upper(values, weights)


def contraction_grid_interval(theta, X, normalized=False, P=None):
    """Enclose every exact-real test value at stored binary64 grid points."""
    from rounding import lambda_max_2x2
    P = _exact_model(theta, P)
    entries = [P[name].interval_eval(X) for name in ("c11", "c12", "c22")]
    result = lambda_max_2x2(*entries)
    if normalized:
        v = P["v"].interval_eval(X)
        # Positive metric construction proves this lower floor independently.
        v = Interval(np.maximum(v.lo, 2.0 * ALPHA), v.hi)
        result = result / v
    return result


# ----------------------- acceptance tests -----------------------

def _numeric_C_entries(theta, X):
    """Reproduce the c11, c12, c22 arrays computed inside cegis_c.lam_max_C.
    Imported lazily to keep bounds.py importable without autograd if needed."""
    from cegis_c import M_entries, jac
    p, q, r, dp, dq, dr = M_entries(theta, X)
    A = jac(X)
    a11 = A[:, 0, 0]; a12 = A[:, 0, 1]; a21 = A[:, 1, 0]; a22 = A[:, 1, 1]
    s11 = 2 * (a11 * p + a21 * q)
    s22 = 2 * (a12 * q + a22 * r)
    s12 = a11 * q + a21 * r + a12 * p + a22 * q
    c11 = dp + s11 + 2 * RATE * p
    c22 = dr + s22 + 2 * RATE * r
    c12 = dq + s12 + 2 * RATE * q
    trM = p + r
    return c11, c12, c22, trM


def test_1b(seed=0, n=200, tol=1e-10, verbose=True):
    """Polynomial entries c11, c12, c22, tr M agree with cegis_c numeric
    code at 200 random (theta, x) pairs, to relative tolerance tol."""
    rng = np.random.default_rng(seed)
    fails = 0
    max_rel = 0.0
    for trial in range(n):
        theta = rng.uniform(-1.0, 1.0, size=(3, 3))
        X = rng.uniform(-BOX, BOX, size=(1, 2))
        P = assemble(theta)
        c11_p = P["c11"].evalv(X)[0]
        c12_p = P["c12"].evalv(X)[0]
        c22_p = P["c22"].evalv(X)[0]
        v_p = P["v"].evalv(X)[0]
        c11_n, c12_n, c22_n, v_n = _numeric_C_entries(theta, X)
        c11_n, c12_n, c22_n, v_n = c11_n[0], c12_n[0], c22_n[0], v_n[0]
        for poly, num in [(c11_p, c11_n), (c12_p, c12_n),
                          (c22_p, c22_n), (v_p, v_n)]:
            denom = max(abs(num), 1.0)
            rel = abs(poly - num) / denom
            if rel > max_rel:
                max_rel = rel
            if rel > tol:
                fails += 1
    if verbose:
        print(f"[1b] {n} pairs, fails={fails}, max relative error={max_rel:.2e}")
    return fails == 0


def _empirical_L_gt(theta, n_grid=401, normalized=True):
    """Empirical Lipschitz estimate by central differences on a fine grid;
    no safety factor. Used as a lower bound for the certified bound to
    beat in test 1c."""
    from cegis_c import lam_max_C, lam_max_C_norm, grid
    fn = lam_max_C_norm if normalized else lam_max_C
    X, _ = grid(n_grid)
    h = 1e-5
    g0 = fn(theta, X)
    gx = (fn(theta, X + np.array([h, 0.0])) - g0) / h
    gy = (fn(theta, X + np.array([0.0, h])) - g0) / h
    return float(np.max(np.sqrt(gx ** 2 + gy ** 2)))


def test_1c_lower_bound(seed=1, n_theta=10, n_samp=20, depth=3, verbose=True):
    """For 10 random theta, with 20 finite-difference grid samplings each
    (n_grid varying), the certified normalized Lipschitz bound is at least
    the empirical estimate (with no safety factor)."""
    rng = np.random.default_rng(seed)
    fails = 0
    samples = []
    for t in range(n_theta):
        theta = rng.uniform(-1.0, 1.0, size=(3, 3))
        L_cert = L_norm(theta, depth=depth)
        for s in range(n_samp):
            n_grid = int(rng.integers(101, 401))
            L_emp = _empirical_L_gt(theta, n_grid=n_grid, normalized=True)
            samples.append((L_cert, L_emp))
            if L_cert + 1e-9 < L_emp:
                fails += 1
    ratios = [c / e if e > 0 else float("inf") for c, e in samples]
    if verbose:
        print(f"[1c lower] {len(samples)} samples, fails={fails}, "
              f"cert/emp ratio: min={min(ratios):.3f} "
              f"median={np.median(ratios):.3f} max={max(ratios):.3f}")
    return fails == 0


def test_1c_monotone(seed=2, n_theta=5, verbose=True):
    """For 5 random theta, the certified bound at depth 4 is at most the
    bound at depth 2: deeper subdivision is tighter (monotone)."""
    rng = np.random.default_rng(seed)
    fails = 0
    for t in range(n_theta):
        theta = rng.uniform(-1.0, 1.0, size=(3, 3))
        L2 = L_norm(theta, depth=2)
        L4 = L_norm(theta, depth=4)
        if L4 > L2 + 1e-12:
            fails += 1
        if verbose:
            print(f"[1c monotone] theta {t}: L(depth=2)={L2:.4f}  "
                  f"L(depth=4)={L4:.4f}  delta={L4 - L2:+.4f}")
    return fails == 0


if __name__ == "__main__":
    ok1b = test_1b()
    ok1c_lo = test_1c_lower_bound()
    ok1c_mo = test_1c_monotone()
    print("\nACCEPTANCE:")
    print(f"  1b: {'PASS' if ok1b else 'FAIL'}")
    print(f"  1c lower bound:  {'PASS' if ok1c_lo else 'FAIL'}")
    print(f"  1c monotonicity: {'PASS' if ok1c_mo else 'FAIL'}")
