"""Replayable analytical constant-metric baseline for the physical swing model.

This is a post hoc fixed-candidate calculation, not an additional trained arm.
The candidate P=[[1,1/2],[1/2,2]] is checked on every previously specified box.
Directed MPFR enclosures preserve the exact physical phase pi/3. All matrix
inequalities and their endpoint/convexity arguments use exact rational algebra.

The displayed target 1/20 and margin 1/10000 are exact decimal rationals. The
shared neural verifier interprets their binary64 encodings as exact reals;
those slightly different targets are also checked explicitly, rather than
silently equated with the decimal rationals.
"""
import argparse
from fractions import Fraction as F
import json
import math
from pathlib import Path

import gmpy2 as g


MODEL_ID = "two_machine_swing_exact_pi_over_3_v1"
METHOD = "fixed_constant_metric_analytical_baseline"
PRECISION = 200
BOXES = (0.25, 0.5, 0.75)
P = ((F(1), F(1, 2)), (F(1, 2), F(2)))
RATE = F(1, 20)
MARGIN = F(1, 10000)
STRONG_MARGIN = F(1, 100)
SHARED_RATE = F.from_float(0.05)
SHARED_MARGIN = F.from_float(0.0001)


def _q(value):
    value = F(value)
    return f"{value.numerator}/{value.denominator}"


def _mp_fraction(value):
    numerator, denominator = value.as_integer_ratio()
    return F(int(numerator), int(denominator))


def _float_bound(value, upward):
    value = F(value)
    result = float(value)
    if not math.isfinite(result):
        raise ArithmeticError("nonfinite rational conversion")
    stored = F.from_float(result)
    if (upward and stored < value) or (not upward and stored > value):
        result = math.nextafter(result, math.inf if upward else -math.inf)
    return result


def cosine_range(B):
    """Enclose cos(delta+pi/3) on [-B,B], including pi and argument errors.

    The two angle endpoints are themselves outward-rounded. Cosine decreases
    throughout this interval, which lies strictly inside (0,pi). The exact
    dyadic MPFR outputs are serialized as rational numbers without a decimal
    formatting step. The right endpoint also supplies a counterexample when
    this particular fixed metric fails the requested rate.
    """
    if isinstance(B, bool) or B not in BOXES:
        raise ValueError("use one of the three fixed application boxes")
    B = F.from_float(float(B))
    with g.context(precision=PRECISION, round=g.RoundDown):
        pi_lo = g.const_pi()
        phase_lo = pi_lo / 3
        angle_lo = phase_lo - g.mpfr(g.mpq(B.numerator, B.denominator))
        endpoint_lo = phase_lo + g.mpfr(g.mpq(B.numerator, B.denominator))
    with g.context(precision=PRECISION, round=g.RoundUp):
        phase_hi = g.const_pi() / 3
        angle_hi = phase_hi + g.mpfr(g.mpq(B.numerator, B.denominator))
    if not (angle_lo > 0 and angle_hi < pi_lo):
        raise ArithmeticError("cosine monotonicity interval was not established")
    with g.context(precision=PRECISION, round=g.RoundDown):
        k_lo = g.cos(angle_hi)
    with g.context(precision=PRECISION, round=g.RoundUp):
        k_hi = g.cos(angle_lo)
        point_k_hi = g.cos(endpoint_lo)
    lower, upper = _mp_fraction(k_lo), _mp_fraction(k_hi)
    # Coarse rational enclosure gives concise independently checkable minors.
    coarse_lo = F((100 * lower).numerator // (100 * lower).denominator, 100)
    scaled_upper = 100 * upper
    coarse_hi = F(-((-scaled_upper.numerator) // scaled_upper.denominator), 100)
    assert coarse_lo <= lower <= upper <= coarse_hi
    return {
        "angle_interval_rational": [_q(_mp_fraction(angle_lo)), _q(_mp_fraction(angle_hi))],
        "monotonicity_interval": "strictly inside (0,pi); cosine is decreasing",
        "mpfr_interval_rational": [_q(lower), _q(upper)],
        "coarse_interval_rational": [_q(coarse_lo), _q(coarse_hi)],
        "right_endpoint_interval_rational": [_q(lower), _q(_mp_fraction(point_k_hi))],
    }


def contraction_matrix(k, rate=RATE):
    """Exact A'P+PA+2*rate*P, where A=[[0,1],[-k,-1/2]]."""
    k, rate = F(k), F(rate)
    return ((2 * rate - k, F(3, 4) + rate - 2 * k),
            (F(3, 4) + rate - 2 * k, -1 + 4 * rate))


def _matrix_json(matrix):
    return [[_q(value) for value in row] for row in matrix]


def _principal_minors(matrix):
    return matrix[0][0], matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def _endpoint_checks(interval, rate, margin):
    rows = []
    for k in interval:
        C = contraction_matrix(k, rate)
        Q = tuple(tuple(-C[i][j] - (margin if i == j else 0) for j in range(2))
                  for i in range(2))
        leading, determinant = _principal_minors(Q)
        rows.append({
            "k_rational": _q(k),
            "negative_shifted_matrix_rational": _matrix_json(Q),
            "leading_principal_minor_rational": _q(leading),
            "determinant_rational": _q(determinant),
            "positive_definite": leading > 0 and determinant > 0,
        })
    return {"rate_rational": _q(rate), "raw_margin_rational": _q(margin),
            "endpoint_checks": rows,
            "certified": all(row["positive_definite"] for row in rows)}


def metric_bounds():
    """Exact rational Loewner bounds; directed upper bound sqrt(b/a)."""
    a, b = F(3, 4), F(9, 4)
    lower = tuple(tuple(P[i][j] - (a if i == j else 0) for j in range(2))
                  for i in range(2))
    upper = tuple(tuple((b if i == j else 0) - P[i][j] for j in range(2))
                  for i in range(2))
    with g.context(precision=PRECISION, round=g.RoundUp):
        prefactor_mp = g.sqrt(3)
    prefactor = _float_bound(_mp_fraction(prefactor_mp), True)
    assert F.from_float(prefactor)**2 >= b/a
    return {
        "a_rational": _q(a), "b_rational": _q(b),
        "lower_shift_principal_minors_rational": [_q(v) for v in _principal_minors(lower)],
        "upper_shift_principal_minors_rational": [_q(v) for v in _principal_minors(upper)],
        "a_lower": float(a), "b_upper": float(b),
        "overshoot_upper": prefactor,
        "overshoot_exact_upper": "sqrt(3)",
    }


def certify_box(B):
    bounds = cosine_range(B)
    interval = tuple(F(value) for value in bounds["coarse_interval_rational"])
    decimal = _endpoint_checks(interval, RATE, MARGIN)
    shared = _endpoint_checks(interval, SHARED_RATE, SHARED_MARGIN)
    stronger = _endpoint_checks(interval, RATE, STRONG_MARGIN)
    certified = shared["certified"] and decimal["certified"]
    result = {
        "box_half_width": float(B),
        "box_rational": [[_q(-F.from_float(float(B))), _q(F.from_float(float(B)))] for _ in range(2)],
        "certified": certified,
        "cosine_bounds": bounds,
        "decimal_target_check": decimal,
        "shared_binary64_target_check": shared,
        "stronger_decimal_margin_check": stronger,
        "proof_rule": "C is affine in k; endpoint negative definiteness extends to their convex hull",
    }
    if certified:
        result["status"] = "certified"
        result["raw_margin_lower_rational_at_decimal_rate"] = _q(STRONG_MARGIN if stronger["certified"] else MARGIN)
        result["target_rate_lower"] = 0.05
        result["matched_raw_margin_lower"] = 0.0001
        result["trajectory_scope"] = "finite-distance guarantees require the stated invariant energy subset and connecting-flow hypotheses"
    else:
        # Failure of a coarse endpoint test alone is inconclusive. Instead prove
        # a positive Rayleigh quotient at the actual point (delta,omega)=(B,0).
        endpoint_k_upper = F(bounds["right_endpoint_interval_rational"][1])
        # A common lower bound proves the violation for both target conventions.
        c11_lower = 2 * min(RATE, SHARED_RATE) - endpoint_k_upper
        if c11_lower <= 0:
            raise ArithmeticError("no actual fixed-candidate violation established")
        result["status"] = "fixed_candidate_violates_requested_rate"
        result["violation"] = {
            "point_rational": [_q(F.from_float(float(B))), "0/1"],
            "direction_rational": ["1/1", "0/1"],
            "rate_conventions_checked": ["exact decimal", "shared binary64"],
            "rayleigh_quotient_lower_rational": _q(c11_lower),
            "rayleigh_quotient_lower": _float_bound(c11_lower, False),
            "interpretation": "this fixed P fails the requested rate; other constant or state-dependent metrics are not excluded",
            "region_scope": "the witness is on the verification-box boundary, outside the strictly interior invariant energy subset; it does not disprove a certificate restricted to that subset",
        }
    return result


def build_report():
    report = {
        "schema_version": 1,
        "status": "pass",
        "method": METHOD,
        "model_id": MODEL_ID,
        "post_hoc": True,
        "training_performed": False,
        "seeds": None,
        "comparison_scope": "analytical fixed-candidate baseline identified after the neural comparison; no matched optimization or timing claim",
        "requested_rate": 0.05,
        "requested_raw_margin": 0.0001,
        "decimal_rate_rational": _q(RATE),
        "decimal_raw_margin_rational": _q(MARGIN),
        "shared_binary64_rate_rational": _q(SHARED_RATE),
        "shared_binary64_raw_margin_rational": _q(SHARED_MARGIN),
        "metric": [[float(v) for v in row] for row in P],
        "metric_rational": _matrix_json(P),
        "metric_positive_principal_minors_rational": [_q(v) for v in _principal_minors(P)],
        "metric_bounds": metric_bounds(),
        "contraction_matrix_at_decimal_rate": {
            "constant_rational": [["1/10", "4/5"], ["4/5", "-4/5"]],
            "k_coefficient_rational": [["-1/1", "-2/1"], ["-2/1", "0/1"]],
            "k": "cos(delta+pi/3)",
        },
        "verification_backend": {"outward_rounded": True, "exact_rational_matrix_checks": True,
                                 "transcendental_backend": "MPFR", "precision_bits": PRECISION},
        "domains": [certify_box(B) for B in BOXES],
        "qualification": "the smallest swing box does not require a learned or state-dependent metric; the two larger fixed-P failures do not establish necessity either",
    }
    assert all(F(v) > 0 for v in report["metric_positive_principal_minors_rational"])
    assert all(F(v) > 0 for key in ("lower_shift_principal_minors_rational", "upper_shift_principal_minors_rational")
               for v in report["metric_bounds"][key])
    assert [row["certified"] for row in report["domains"]] == [True, False, False]
    return report


def validate_report(data):
    """Replay every directed transcendental enclosure and exact matrix check.

    Canonical complete comparison also rejects omitted/altered domain records,
    target changes, fabricated training metadata, and weakened qualifications.
    The independent mathematical tests avoid relying on this replay alone.
    """
    if not isinstance(data, dict) or data != build_report():
        raise ValueError("constant-metric report differs from exact/directed proof replay")
    return {"status": "pass", "domains": len(data["domains"]),
            "certified_domains": sum(row["certified"] for row in data["domains"]),
            "fixed_candidate_violations": sum(row["status"] == "fixed_candidate_violates_requested_rate"
                                              for row in data["domains"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true",
                        help="replay the existing saved JSON without modifying it")
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent / "results/application_constant_baseline.json"
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
