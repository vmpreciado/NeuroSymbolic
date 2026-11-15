"""Exact fixed constant-metric certificate for the polynomial benchmark.

This candidate was identified after the learning experiments. It is neither a
trained comparison arm nor an optimized search over constant metrics. For
f(x)=(-x1,-x2+x1**2) on [-2,2]**2, P=diag(12,1) certifies the stated rate 2/5
and its slightly larger stored binary64 value. All matrix calculations use
exact fractions. Affinity in x1 extends the two endpoint checks to the box;
x2 does not enter the Jacobian or the constant-metric contraction matrix.

The independently selected raw margin 1/20 and normalized margin 1/260 are
proved lower bounds, not the training margins of the learned candidates.
"""
import argparse
from fractions import Fraction as F
import json
import math
from pathlib import Path


MODEL_ID = "triangular_polynomial_exact_v1"
METHOD = "fixed_constant_metric_analytical_baseline"
P = ((F(12), F(0)), (F(0), F(1)))
BOX = F(2)
RATE = F(2, 5)
SHARED_RATE = F.from_float(0.4)
MARGIN = F(1, 20)
TRACE = sum(P[i][i] for i in range(2))


def _q(value):
    value = F(value)
    return f"{value.numerator}/{value.denominator}"


def _matrix_json(matrix):
    return [[_q(value) for value in row] for row in matrix]


def _principal_minors(matrix):
    return (matrix[0][0],
            matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])


def contraction_matrix(x1, rate=RATE):
    """Return exact A'P+PA+2*rate*P for A=[[-1,0],[2*x1,-1]]."""
    x1, rate = F(x1), F(rate)
    return ((24 * (rate - 1), 2 * x1), (2 * x1, 2 * (rate - 1)))


def _endpoint_checks(rate):
    rows = []
    for x1 in (-BOX, BOX):
        C = contraction_matrix(x1, rate)
        negative = tuple(tuple(-C[i][j] for j in range(2)) for i in range(2))
        shifted = tuple(tuple(negative[i][j] - (MARGIN if i == j else 0)
                              for j in range(2)) for i in range(2))
        leading, determinant = _principal_minors(negative)
        shifted_leading, shifted_determinant = _principal_minors(shifted)
        rows.append({
            "x1_rational": _q(x1),
            "contraction_matrix_rational": _matrix_json(C),
            "negative_matrix_rational": _matrix_json(negative),
            "leading_principal_minor_rational": _q(leading),
            "determinant_rational": _q(determinant),
            "negative_shifted_matrix_rational": _matrix_json(shifted),
            "shifted_leading_principal_minor_rational": _q(shifted_leading),
            "shifted_determinant_rational": _q(shifted_determinant),
            "positive_definite": leading > 0 and determinant > 0,
            "shifted_positive_definite": shifted_leading > 0 and shifted_determinant > 0,
        })
    return {
        "rate_rational": _q(rate),
        "raw_margin_rational": _q(MARGIN),
        "normalized_margin_rational": _q(MARGIN / TRACE),
        "endpoint_checks": rows,
        "certified": all(row["positive_definite"] and row["shifted_positive_definite"]
                         for row in rows),
    }


def metric_bounds():
    """Exact diagonal Loewner bounds and a rationally checked sqrt(12) bound."""
    a, b = F(1), F(12)
    lower_shift = tuple(P[i][i] - a for i in range(2))
    upper_shift = tuple(b - P[i][i] for i in range(2))
    # The nominal square root is only a starting value. Its exact rational
    # square decides whether an upward step is necessary, and verifies the
    # delivered binary64 bound independently of the square-root rounding.
    prefactor = math.sqrt(float(b / a))
    if not math.isfinite(prefactor) or prefactor <= 0:
        raise ArithmeticError("invalid square-root starting approximation")
    while F.from_float(prefactor) ** 2 < b / a:
        prefactor = math.nextafter(prefactor, math.inf)
    if not all(value >= 0 for value in lower_shift + upper_shift):
        raise ArithmeticError("invalid diagonal metric bounds")
    return {
        "a_rational": _q(a), "b_rational": _q(b),
        "a_lower": float(a), "b_upper": float(b),
        "lower_shift_diagonal_rational": [_q(value) for value in lower_shift],
        "upper_shift_diagonal_rational": [_q(value) for value in upper_shift],
        "overshoot_exact_upper": "sqrt(12)",
        "overshoot_upper": prefactor,
        "overshoot_upper_rational": _q(F.from_float(prefactor)),
        "overshoot_square_at_least_b_over_a": F.from_float(prefactor) ** 2 >= b / a,
    }


def build_report():
    decimal = _endpoint_checks(RATE)
    shared = _endpoint_checks(SHARED_RATE)
    report = {
        "schema_version": 1,
        "status": "pass",
        "method": METHOD,
        "model_id": MODEL_ID,
        "certified": decimal["certified"] and shared["certified"],
        "post_hoc": True,
        "training_performed": False,
        "seeds": None,
        "comparison_scope": "fixed analytical candidate identified after the polynomial learning experiments; no optimized constant-metric search, matched training, or timing comparison",
        "domain": {"box_half_width": float(BOX),
                   "box_rational": [[_q(-BOX), _q(BOX)] for _ in range(2)],
                   "state_dimension": 2},
        "dynamics": ["-x1", "-x2+x1^2"],
        "jacobian": [["-1", "0"], ["2*x1", "-1"]],
        "requested_rate": 0.4,
        "decimal_rate_rational": _q(RATE),
        "shared_binary64_rate_rational": _q(SHARED_RATE),
        "raw_margin_lower_rational": _q(MARGIN),
        "normalized_margin_lower_rational": _q(MARGIN / TRACE),
        "margin_scope": "independently selected safe analytical margins; not the learned candidates' training margins",
        "metric": [[float(value) for value in row] for row in P],
        "metric_rational": _matrix_json(P),
        "metric_trace_rational": _q(TRACE),
        "metric_positive_principal_minors_rational": [_q(value) for value in _principal_minors(P)],
        "metric_bounds": metric_bounds(),
        "contraction_matrix_at_decimal_rate": {
            "constant_rational": _matrix_json(contraction_matrix(0, RATE)),
            "x1_coefficient_rational": [["0/1", "2/1"], ["2/1", "0/1"]],
            "independent_of_x2": True,
        },
        "decimal_target_check": decimal,
        "shared_binary64_target_check": shared,
        "proof_rule": "C is affine in x1 and independent of x2; the positive-definite cone is convex, so exact Sylvester checks at x1=-2 and x1=2 prove C(x) strictly below -(1/20)I throughout the box",
        "normalization_rule": "trace(P)=13, so lambda_max(C)/trace(P) is strictly below -1/260",
        "verification_backend": {
            "exact_rational_matrix_checks": True,
            "matrix_arithmetic": "Python fractions.Fraction; no floating-point matrix or eigenvalue tests",
            "prefactor_bound": "binary64 square-root starting value, corrected upward and checked by exact rational squaring",
        },
        "qualification": "a learned or state-dependent metric is unnecessary to certify this polynomial benchmark at the target rate; the learning experiments diagnose synthesis and verification behavior, not a need for neural expressiveness",
        "trajectory_scope": "the verification box is not forward invariant; finite-distance guarantees require the manuscript's connecting-flow and domain hypotheses",
    }
    if not report["certified"] or not all(F(value) > 0 for value in report["metric_positive_principal_minors_rational"]):
        raise ArithmeticError("fixed polynomial metric did not certify the requested rate and margin")
    return report


def validate_report(data):
    """Replay the full exact proof and reject any changed evidence or scope.

    Comparing canonical JSON also distinguishes booleans from integers and
    refuses nonfinite values; Python container equality alone would not.
    """
    try:
        actual = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected = json.dumps(build_report(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("polynomial constant-metric report is not finite JSON") from exc
    if not isinstance(data, dict) or actual != expected:
        raise ValueError("polynomial constant-metric report differs from exact proof replay")
    return {"status": "pass", "certified": True, "target_conventions": 2,
            "endpoint_checks_per_convention": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true",
                        help="replay the existing saved JSON without modifying it")
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent / "results/polynomial_constant_baseline.json"
    if args.validate:
        data = json.loads(destination.read_text())
    else:
        data = build_report()
        validate_report(data)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(json.dumps(validate_report(data), sort_keys=True))


if __name__ == "__main__":
    main()
