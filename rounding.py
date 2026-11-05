"""Outward binary64 interval arithmetic for the certificate path.

Stored binary64 model parameters and grid nodes denote exact real numbers.
Every elementary binary64 operation is bracketed with ``nextafter`` in both
directions. This relies on IEEE 754 round-to-nearest arithmetic, correctly
rounded binary64 square root, and gradual underflow; ``self_test`` rejects
unsupported rounding/subnormal environments. Arithmetic overflow and invalid
interval operations fail closed.

NumPy's transcendental accuracy is not trusted. Small endpoint batches use
MPFR directed rounding. Large batches use piecewise linear interpolation of
directed MPFR node bounds, enlarged by an analytic interpolation remainder.
For step h, |tanh''| <= 2 gives h**2/4 and |sin''|,|cos''| <= 1 give h**2/8.
Thus the table is a proven enclosure, rather than an empirical error estimate.
"""
from functools import lru_cache
from fractions import Fraction
import math

import gmpy2
import numpy as np

_NEG_INF = np.float64(-np.inf)
_POS_INF = np.float64(np.inf)
_PRECISION = 160
_SMALL_BATCH = 4096
_TABLE_STEP = 2.0 ** -10
_TABLE_LIMIT = 16.0


def _finite(value):
    arr = np.asarray(value, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ArithmeticError("nonfinite value in outward interval arithmetic")
    return arr


def _direct(value, direction):
    value = _finite(value)
    with np.errstate(all="ignore"):
        out = np.nextafter(value, direction)
    return _finite(out)


def down(value):
    return _direct(value, _NEG_INF)


def up(value):
    return _direct(value, _POS_INF)


def _binary(a, b, operation, direction):
    with np.errstate(all="ignore"):
        result = operation(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return _direct(result, direction)


def _scalar(value):
    result = np.asarray(value)
    if result.ndim:
        raise ValueError("scalar directed helper received an array")
    return float(result)


def add_up(a, b): return _scalar(_binary(a, b, np.add, _POS_INF))
def add_down(a, b): return _scalar(_binary(a, b, np.add, _NEG_INF))
def sub_up(a, b): return _scalar(_binary(a, b, np.subtract, _POS_INF))
def sub_down(a, b): return _scalar(_binary(a, b, np.subtract, _NEG_INF))
def mul_up(a, b): return _scalar(_binary(a, b, np.multiply, _POS_INF))
def mul_down(a, b): return _scalar(_binary(a, b, np.multiply, _NEG_INF))
def div_up(a, b):
    if b == 0: raise ZeroDivisionError("zero divisor in directed arithmetic")
    return _scalar(_binary(a, b, np.divide, _POS_INF))
def div_down(a, b):
    if b == 0: raise ZeroDivisionError("zero divisor in directed arithmetic")
    return _scalar(_binary(a, b, np.divide, _NEG_INF))


def float_down(value):
    """Convert a Fraction, MPFR number, or finite scalar toward -infinity."""
    result = float(value)
    if not math.isfinite(result):
        raise ArithmeticError("nonfinite outward conversion")
    exact_result = Fraction.from_float(result) if isinstance(value, Fraction) else gmpy2.mpq(result)
    if exact_result > value:
        result = float(np.nextafter(result, -np.inf))
    if not math.isfinite(result):
        raise ArithmeticError("overflow in outward conversion")
    return result


def float_up(value):
    """Convert a Fraction, MPFR number, or finite scalar toward +infinity."""
    result = float(value)
    if not math.isfinite(result):
        raise ArithmeticError("nonfinite outward conversion")
    exact_result = Fraction.from_float(result) if isinstance(value, Fraction) else gmpy2.mpq(result)
    if exact_result < value:
        result = float(np.nextafter(result, np.inf))
    if not math.isfinite(result):
        raise ArithmeticError("overflow in outward conversion")
    return result


def _mpfr_point(value, name):
    """Two directed MPFR evaluations with directed binary64 conversion."""
    value = float(value)
    if not math.isfinite(value):
        raise ArithmeticError("nonfinite transcendental input")
    with gmpy2.context(precision=_PRECISION, round=gmpy2.RoundDown):
        lower = getattr(gmpy2, name)(gmpy2.mpfr(value))
        lo = float_down(lower)
    with gmpy2.context(precision=_PRECISION, round=gmpy2.RoundUp):
        upper = getattr(gmpy2, name)(gmpy2.mpfr(value))
        hi = float_up(upper)
    return lo, hi


def sqrt_up(value):
    if value < 0: raise ValueError("negative square root")
    return _mpfr_point(value, "sqrt")[1]


def sqrt_down(value):
    if value < 0: raise ValueError("negative square root")
    return _mpfr_point(value, "sqrt")[0]


class Interval:
    """Finite binary64 endpoints enclosing all intermediate roundoff.

    ``Interval(x)`` treats the stored binary64 value x as an exact point.
    For non-binary inputs, use float_down/float_up before construction.
    ``sqrt`` takes the intersection with its real domain [0,infinity).
    """
    __slots__ = ("lo", "hi", "_point")
    __array_priority__ = 10000

    def __init__(self, lo, hi=None):
        self._point = hi is None
        self.lo = _finite(lo)
        self.hi = self.lo if hi is None else _finite(hi)
        self.lo, self.hi = np.broadcast_arrays(self.lo, self.hi)
        if np.any(self.lo > self.hi):
            raise ValueError("reversed interval endpoints")

    def __repr__(self):
        return f"Interval({self.lo!r}, {self.hi!r})"

    def _scalar_point(self):
        if self.lo.ndim == 0 and self.lo == self.hi:
            return float(self.lo)
        return None

    def __add__(self, other):
        other = as_interval(other)
        if other._scalar_point() == 0.0: return self
        if self._scalar_point() == 0.0: return other
        return Interval(_binary(self.lo, other.lo, np.add, _NEG_INF),
                        _binary(self.hi, other.hi, np.add, _POS_INF))
    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-as_interval(other))

    def __rsub__(self, other):
        return as_interval(other) - self

    def __mul__(self, other):
        other = as_interval(other)
        scalar = other._scalar_point()
        if scalar is None:
            scalar = self._scalar_point()
            if scalar is not None: return other * scalar
        else:
            if scalar == 0.0: return Interval(np.zeros_like(self.lo))
            if scalar == 1.0: return self
            if scalar == -1.0: return -self
            a, b = (self.lo, self.hi) if scalar > 0 else (self.hi, self.lo)
            return Interval(_binary(a, scalar, np.multiply, _NEG_INF),
                            _binary(b, scalar, np.multiply, _POS_INF))
        with np.errstate(all="ignore"):
            products = [self.lo*other.lo, self.lo*other.hi,
                        self.hi*other.lo, self.hi*other.hi]
        lo = np.minimum.reduce(products)
        hi = np.maximum.reduce(products)
        return Interval(down(lo), up(hi))
    __rmul__ = __mul__

    def __truediv__(self, other):
        other = as_interval(other)
        if np.any((other.lo <= 0) & (other.hi >= 0)):
            raise ZeroDivisionError("interval divisor contains zero")
        scalar = other._scalar_point()
        if scalar == 1.0: return self
        if scalar == -1.0: return -self
        with np.errstate(all="ignore"):
            values = [self.lo/other.lo, self.lo/other.hi,
                      self.hi/other.lo, self.hi/other.hi]
        return Interval(down(np.minimum.reduce(values)), up(np.maximum.reduce(values)))

    def __rtruediv__(self, other):
        return as_interval(other) / self

    def __pow__(self, power):
        if not isinstance(power, (int, np.integer)) or power < 0:
            raise ValueError("interval powers must be nonnegative integers")
        if power == 0: return Interval(np.ones_like(self.lo))
        if power == 1: return self
        if power == 2: return self.square()
        result = Interval(1.0)
        factor = self
        while power:
            if power & 1: result = result * factor
            power //= 2
            if power: factor = factor.square()
        return result

    def square(self):
        with np.errstate(all="ignore"):
            a, b = self.lo*self.lo, self.hi*self.hi
        lower = np.where((self.lo <= 0) & (self.hi >= 0), 0.0,
                         np.maximum(0.0, down(np.minimum(a, b))))
        upper = up(np.maximum(a, b))
        return Interval(lower, upper)

    def sqrt(self):
        if np.any(self.hi < 0): raise ValueError("negative interval square root")
        with np.errstate(all="ignore"):
            lower = np.sqrt(np.maximum(self.lo, 0.0))
            upper = np.sqrt(np.maximum(self.hi, 0.0))
        return Interval(np.maximum(0.0, down(lower)), up(upper))

    def absmax(self):
        return np.maximum(np.abs(self.lo), np.abs(self.hi))

    def maximum(self, other):
        other = as_interval(other)
        return Interval(np.maximum(self.lo, other.lo), np.maximum(self.hi, other.hi))

    def minimum(self, other):
        other = as_interval(other)
        return Interval(np.minimum(self.lo, other.lo), np.minimum(self.hi, other.hi))

    def tanh(self):
        if self._point:
            lower, upper = _point_function(self.lo, "tanh")
            return Interval(np.maximum(-1.0, lower), np.minimum(1.0, upper))
        if self.lo.size > _SMALL_BATCH:
            # A single table evaluation suffices for a narrow grid-point
            # interval: |tanh(x)-tanh(center)| <= |x-center|. The outward
            # radius, rather than an assumed libm error, covers every input.
            center = self.lo*0.5 + self.hi*0.5
            radius = np.maximum(_binary(center, self.lo, np.subtract, _POS_INF),
                                _binary(self.hi, center, np.subtract, _POS_INF))
            lower, upper = _point_function(center, "tanh")
            return Interval(np.maximum(-1.0, _binary(lower, radius, np.subtract, _NEG_INF)),
                            np.minimum(1.0, _binary(upper, radius, np.add, _POS_INF)))
        lower, _ = _point_function(self.lo, "tanh")
        _, upper = _point_function(self.hi, "tanh")
        return Interval(np.maximum(-1.0, lower), np.minimum(1.0, upper))

    def _trig(self, name):
        if self._point:
            lo, hi = _point_function(self.lo, name)
            return Interval(lo, hi)
        # Any binary64 center suffices: a directed radius contains both ends.
        center = self.lo*0.5 + self.hi*0.5
        radius = np.maximum(_binary(center, self.lo, np.subtract, _POS_INF),
                            _binary(self.hi, center, np.subtract, _POS_INF))
        lo, hi = _point_function(center, name)
        return Interval(np.maximum(-1.0, _binary(lo, radius, np.subtract, _NEG_INF)),
                        np.minimum(1.0, _binary(hi, radius, np.add, _POS_INF)))

    def sin(self): return self._trig("sin")
    def cos(self): return self._trig("cos")


Iv = Interval


def point(value):
    return Interval(value)


def as_interval(value):
    return value if isinstance(value, Interval) else Interval(value)


@lru_cache(maxsize=3)
def _function_table(name):
    count = int(2*_TABLE_LIMIT/_TABLE_STEP)
    nodes = -_TABLE_LIMIT + np.arange(count+1, dtype=np.float64)*_TABLE_STEP
    lower = np.empty(count+1); upper = np.empty(count+1)
    for j, x in enumerate(nodes):
        lower[j], upper[j] = _mpfr_point(float(x), name)
    for a in (nodes, lower, upper): a.flags.writeable = False
    return nodes, lower, upper


@lru_cache(maxsize=3)
def _function_slopes(name):
    """Outward slopes of the chords between directed MPFR node values."""
    _, lower, upper = _function_table(name)
    slopes = (Interval(lower[1:], upper[1:])-
              Interval(lower[:-1], upper[:-1]))/_TABLE_STEP
    slopes.lo.flags.writeable = False
    slopes.hi.flags.writeable = False
    return slopes.lo, slopes.hi


def _point_function(value, name):
    value = _finite(value)
    if value.size <= _SMALL_BATCH:
        lo = np.empty_like(value); hi = np.empty_like(value)
        for index in np.ndindex(value.shape):
            lo[index], hi[index] = _mpfr_point(float(value[index]), name)
        return lo, hi
    nodes, node_lo, node_hi = _function_table(name)
    # Outside the tabulated domain, use global range bounds (and monotonic
    # tanh tail bounds). These are sound even for very large finite inputs.
    lo = np.full(value.shape, -1.0); hi = np.full(value.shape, 1.0)
    if name == "tanh":
        lo[value >= _TABLE_LIMIT] = node_lo[-1]
        hi[value <= -_TABLE_LIMIT] = node_hi[0]
    inside = (value >= -_TABLE_LIMIT) & (value <= _TABLE_LIMIT)
    x = value[inside]
    if not x.size: return lo, hi
    index = np.floor((x+_TABLE_LIMIT)/_TABLE_STEP).astype(np.int64)
    index = np.clip(index, 0, len(nodes)-2)
    # Rounded index arithmetic can differ by one at a cell boundary. Comparing
    # with exactly represented dyadic nodes repairs it before interpolation.
    index = np.where(x < nodes[index], index-1, index)
    index = np.where(x > nodes[index+1], index+1, index)
    index = np.clip(index, 0, len(nodes)-2)
    left = nodes[index]
    if np.any(x < left) or np.any(x > nodes[index+1]):
        raise ArithmeticError("transcendental interpolation cell does not contain input")
    displacement = Interval(x)-left
    displacement = Interval(np.maximum(0.0, displacement.lo),
                            np.minimum(_TABLE_STEP, displacement.hi))
    a = Interval(node_lo[index], node_hi[index])
    slope_lo, slope_hi = _function_slopes(name)
    slope = Interval(slope_lo[index], slope_hi[index])
    interpolation = a + slope*displacement
    remainder = _TABLE_STEP*_TABLE_STEP/(4.0 if name == "tanh" else 8.0)
    enclosure = interpolation + Interval(-remainder, remainder)
    lo[inside] = np.maximum(-1.0, enclosure.lo)
    hi[inside] = np.minimum(1.0, enclosure.hi)
    return lo, hi


def lambda_max_2x2(c11, c12, c22):
    """Enclose the largest real eigenvalue of a symmetric 2 by 2 matrix."""
    a, b, c = as_interval(c11), as_interval(c12), as_interval(c22)
    center = (a+c)*0.5
    difference = (a-c)*0.5
    return center + (difference.square()+b.square()).sqrt()


def grid_radius(nodes, dimension=2):
    """Outward Cartesian-grid cover radius from maximum stored node spacing.

    The caller must ensure that the first/last nodes are its domain endpoints.
    Using the first spacing alone is unsafe for rounded linspace nodes.
    """
    nodes = _finite(nodes)
    if nodes.ndim != 1 or len(nodes) < 2:
        raise ValueError("grid nodes must be a one-dimensional array of length >= 2")
    if not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError("grid dimension must be positive")
    if np.any(nodes[1:] <= nodes[:-1]):
        raise ValueError("grid nodes must increase strictly")
    gap = float(np.max(_binary(nodes[1:], nodes[:-1], np.subtract, _POS_INF)))
    return mul_up(mul_up(gap, 0.5), sqrt_up(float_up(Fraction(int(dimension)))))


def self_test():
    """Reject unsupported elementary rounding/subnormal environments."""
    if np.dtype(np.float64).itemsize != 8 or np.finfo(np.float64).nmant != 52:
        raise ArithmeticError("IEEE binary64 is required")
    one = np.float64(1.0); half_ulp = np.float64(2.0**-53)
    if one+half_ulp != one or -one-half_ulp != -one or one+3*half_ulp != one+np.float64(2.0**-51):
        raise ArithmeticError("round-to-nearest, ties-to-even arithmetic is required")
    subnormal = np.nextafter(np.float64(0), np.float64(1))
    if subnormal <= 0 or subnormal*one != subnormal or np.float64(np.finfo(np.float64).tiny)*0.5 <= 0:
        raise ArithmeticError("gradual underflow is required")
    for x in (np.nextafter(0.0, 1.0), 0.5, 2.0, 1e200):
        result = Interval(x).sqrt()
        lo, hi = _mpfr_point(x, "sqrt")
        if float(result.lo) > lo or float(result.hi) < hi:
            raise ArithmeticError("binary64 square-root rounding check failed")
    return True


self_tests = self_test


def backend_info():
    self_test()
    return {"outward_rounded": True,
            "backend": "binary64-nextafter with MPFR directed transcendental enclosures",
            "gmpy2_version": gmpy2.version(), "mpfr_version": gmpy2.mpfr_version(),
            "parameter_semantics": "stored binary64 parameters and grid nodes denote exact real values",
            "elementary_arithmetic": "IEEE 754 binary64 round-to-nearest ties-to-even, correctly rounded sqrt, gradual underflow",
            "overflow_policy": "fail closed",
            "transcendental_precision_bits": _PRECISION,
            "large_batch_transcendental_method": "directed-MPFR node interpolation plus analytic second-derivative remainder",
            "interpolation_step": _TABLE_STEP,
            "interpolation_domain": [-_TABLE_LIMIT, _TABLE_LIMIT]}
