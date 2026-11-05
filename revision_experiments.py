"""Controlled revision experiments, using the unchanged outward grid oracle.

Run the production study with ``python revision_experiments.py``.  ``--pilot``
uses separate output names and smaller learner budgets; it is not paper evidence.
The constructed scale chase is an existence illustration, not Adam behavior.
"""
from __future__ import annotations

# Set these before importing numpy/autograd; the integrated runner does likewise.
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bounds
import cegis_c
import experiments
from rounding import (add_down, add_up, backend_info, div_down, div_up,
                      float_up, mul_down, mul_up, sqrt_up, sub_down)

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Protocol:
    n_grid: int = 2001
    max_rounds: int = 12
    training_steps: int = 500
    learning_rate: float = 0.02
    initial_samples: int = 500
    base_margin: float = 0.05
    gamma: float = 1.3
    k_worst: int | str = 60
    perturbations: int = 5
    perturbation_std: float = 0.08
    depth: int = 3
    radius: float | None = None


def radius_rule() -> float:
    """Predeclared radius, independent of outcome files and fitted candidates."""
    return 4.0 * float(np.linalg.norm(experiments.identity_init()))


def initial_state(seed: int, config: Protocol):
    rng = np.random.RandomState(seed)
    data = rng.uniform(-cegis_c.BOX, cegis_c.BOX, (config.initial_samples, 2))
    return experiments.identity_init(), data, rng


def project_ball(theta, radius):
    if radius is None:
        return np.asarray(theta)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("projection radius must be positive and finite")
    values = np.asarray(theta, dtype=float)
    norm2 = exact_norm_squared(values)
    radius2 = Fraction.from_float(float(radius)) ** 2
    if norm2 <= radius2:
        return values
    # Use an upper enclosure for the real norm, then recheck the ACTUAL rounded
    # vector.  Inward scalar rounding alone would not prove the parameter bound.
    norm_upper = sqrt_up(float_up(norm2))
    projected = values * np.nextafter(radius / norm_upper, 0.0)
    while exact_norm_squared(projected) > radius2:
        projected = np.nextafter(projected, 0.0)
    return projected


def exact_norm_squared(theta):
    values = np.asarray(theta, dtype=float)
    if not np.isfinite(values).all():
        raise ArithmeticError("nonfinite parameter vector")
    return sum((Fraction.from_float(float(value)) ** 2 for value in values.flat),
               Fraction(0))


def learn(theta, data, eps, normalized, config):
    """Adam with moments reset each round; optional projection at EVERY step.

    Projection is a practical bounded-parameter variant.  Neither zero empirical
    loss nor the exact-feasibility hypothesis of a termination theorem is assumed.
    """
    theta = project_ball(theta, config.radius).copy()
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    max_norm, projections, updates = float(np.linalg.norm(theta)), 0, 0
    max_norm2 = exact_norm_squared(theta) if config.radius is not None else None
    for step in range(1, config.training_steps + 1):
        grad = np.asarray(cegis_c.dloss(theta, data, eps, normalized))
        if not np.isfinite(grad).all():
            raise ArithmeticError("nonfinite learner gradient")
        m = 0.9 * m + (1.0 - 0.9) * grad
        v = 0.999 * v + (1.0 - 0.999) * grad ** 2
        candidate = theta - config.learning_rate * (m / (1 - 0.9 ** step)) / (
            np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        if not np.isfinite(candidate).all():
            raise ArithmeticError("nonfinite learner update")
        theta = project_ball(candidate, config.radius)
        projections += int(config.radius is not None and not np.array_equal(candidate, theta))
        updates = step
        max_norm = max(max_norm, float(np.linalg.norm(theta)))
        if config.radius is not None:
            current_norm2 = exact_norm_squared(theta)
            if current_norm2 > Fraction.from_float(config.radius) ** 2:
                raise AssertionError("exact stored parameter norm exceeds chosen radius")
            max_norm2 = max(max_norm2, current_norm2)
        if float(cegis_c.loss(theta, data, eps, normalized)) == 0.0:
            break
    return theta, {"adam_updates": updates, "projection_steps": projections,
                   "max_post_projection_norm": max_norm,
                   "max_parameter_norm_squared_exact": str(max_norm2) if max_norm2 is not None else None,
                   "exact_parameter_ball_enforced": config.radius is not None,
                   "empirical_loss": float(cegis_c.loss(theta, data, eps, normalized))}


def select_margin_points(verification, k_worst):
    values = np.asarray(verification["g"])
    if k_worst == "all":
        return np.flatnonzero(np.asarray(verification["viol"]))
    k = min(int(k_worst), len(values))
    if k <= 0:
        raise ValueError("k_worst must be positive or 'all'")
    # Match the historical top-k selection, including diagnostic margin points.
    return np.argsort(-values)[:k]


def quantitative_outputs(theta, verification, normalized, depth):
    if not verification["certified"]:
        return {}
    if not verification["grid_enclosed"] or verification["uniform_slack"] < 0:
        raise AssertionError("positive result without outward grid evidence")
    slack = float(verification["uniform_slack"])
    a = float(cegis_c.ALPHA)
    b = bounds.b_upper_bound(theta, depth=depth)
    # Use the universally valid trace floor, recording the conversion explicitly.
    eta = max(0.0, mul_down(2.0 * a, slack)) if normalized else slack
    rate = max(float(cegis_c.RATE),
               add_down(float(cegis_c.RATE), div_down(eta, mul_up(2.0, b))))
    return {"a": a, "b_upper": b, "raw_margin_lower": eta,
            "raw_margin_method": "2*alpha*normalized_uniform_slack" if normalized
                                 else "raw_outward_grid_slack",
            "rate_lower": rate, "overshoot_upper": sqrt_up(div_up(b, a)),
            "metric_bounds_outward_rounded": True}


def summarize(runs):
    values = list(runs.values())
    success = [r["iters_to_cert"] for r in values if r["certified"]]
    return {"n": len(values), "certified": sum(r["certified"] for r in values),
            "diverged": sum(r["diverged"] for r in values),
            "stalled": sum(r["stalled"] for r in values),
            "numerical_failure": sum(r.get("numerical_failure", False) for r in values),
            "median_iters_to_cert": float(np.median(success)) if success else None,
            "max_parameter_norm": max((max(t["parameter_norm"] for t in r["trace"])
                                        for r in values), default=None),
            "median_total_seconds": float(np.median([r["elapsed_s"] for r in values]))
                                    if values else None}


def run_controlled(seed, normalized, coupled, config=Protocol()):
    theta, data, rng = initial_state(seed, config)
    initial_dataset_sha256 = hashlib.sha256(data.astype('<f8').tobytes()).hexdigest()
    initial_theta_sha256 = hashlib.sha256(theta.astype('<f8').tobytes()).hexdigest()
    if config.n_grid < 2 or config.max_rounds < 0 or config.training_steps < 1:
        raise ValueError("invalid experiment budget")
    if config.radius is not None and exact_norm_squared(theta) > Fraction.from_float(config.radius) ** 2:
        raise ValueError("predeclared ball excludes the common initial metric")
    trace, timings, eps = [], [], config.base_margin
    train_s, learner = 0.0, {"adam_updates": 0, "projection_steps": 0,
                           "max_post_projection_norm": float(np.linalg.norm(theta)),
                           "max_parameter_norm_squared_exact": str(exact_norm_squared(theta)) if config.radius is not None else None,
                           "exact_parameter_ball_enforced": config.radius is not None,
                           "empirical_loss": None}
    started, failure, verification = time.perf_counter(), None, None
    for iteration in range(config.max_rounds + 1):
        before = time.perf_counter()
        try:
            verification = cegis_c.verify(theta, config.n_grid, normalized=normalized,
                                           certified=True, cert_depth=config.depth)
        except (ArithmeticError, ValueError, FloatingPointError) as exc:
            failure = f"verification failed closed: {type(exc).__name__}: {exc}"
            break
        verify_s = time.perf_counter() - before
        outward = config.n_grid ** 2 if verification["grid_enclosed"] else 0
        nominal = config.n_grid ** 2
        center = np.array([[0.0, 0.0]])
        m11, _, m22, *_ = cegis_c.M_entries(theta, center)
        trace.append({"iter": iteration, "L": float(verification["L"]),
                      "tau": float(verification["tau"]),
                      "required_margin_upper": -float(verification["margin"]),
                      "worst": float(verification["worst"]),
                      "gap": float(verification["worst"] - verification["margin"]),
                      "uniform_slack": float(verification["uniform_slack"]),
                      "slack_is_certified": bool(verification["grid_enclosed"]),
                      "grid_enclosed": bool(verification["grid_enclosed"]),
                      "outward_rounded": True,
                      "grid_values_kind": verification["grid_values_kind"],
                      "viol": int(np.count_nonzero(verification["viol"])),
                      "eps": float(eps), "dataset_size": len(data),
                      "parameter_norm": float(np.linalg.norm(theta)),
                      "metric_trace_at_origin": float(m11[0] + m22[0]),
                      "theta": theta.tolist(), **learner})
        timings.append({"iter": iteration, "train_s": train_s,
                        "verify_including_bounds_s": verify_s,
                        "oracle_evaluations": nominal + outward,
                        "nominal_grid_evaluations": nominal,
                        "outward_grid_evaluations": outward})
        if verification["certified"] or iteration == config.max_rounds:
            break
        if coupled:
            eps = max(config.base_margin, config.gamma * verification["L"] *
                      verification["tau"])
        indices = select_margin_points(verification, config.k_worst)
        points = np.asarray(verification["X"])[indices]
        perturbed = np.clip(np.repeat(points, config.perturbations, axis=0) +
                            config.perturbation_std * rng.randn(
                                config.perturbations * len(points), 2),
                            -cegis_c.BOX, cegis_c.BOX)
        data = np.vstack([data, points, perturbed])
        before = time.perf_counter()
        try:
            theta, learner = learn(theta, data, eps, normalized, config)
        except (ArithmeticError, ValueError, FloatingPointError) as exc:
            failure = f"learner failed closed: {type(exc).__name__}: {exc}"
            break
        train_s = time.perf_counter() - before
    certified = failure is None and verification is not None and bool(verification["certified"])
    if not trace:
        raise RuntimeError(f"initial verification failed: {failure}")
    verdict = experiments.classify(trace, certified)
    if failure:
        verdict.update(diverged=False, stalled=False)
    verdict.update(seed=seed, normalized=normalized, coupled=coupled,
                   protocol=asdict(config), theta=theta.tolist(),
                   initial_dataset_sha256=initial_dataset_sha256,
                   initial_theta_sha256=initial_theta_sha256,
                   numerical_failure=failure is not None,
                   failure_reason=failure, elapsed_s=time.perf_counter() - started,
                   margin_units="lambda_max(C)/trace(M)" if normalized else "lambda_max(C)",
                   metric_scale_diagnostic="trace(M(0)); not a scale/shape decomposition")
    if certified:
        verdict.update(quantitative_outputs(theta, verification, normalized, config.depth))
    return verdict, timings


def scale_polynomials(model, scale):
    """Scale every exact coefficient; do not reinterpret a fixed-floor theta."""
    if isinstance(model, bounds.ExactPoly2):
        return model * scale
    if isinstance(model, dict):
        return {key: scale_polynomials(value, scale) for key, value in model.items()}
    if isinstance(model, (tuple, list)):
        return [scale_polynomials(value, scale) for value in model]
    raise TypeError("unexpected assembled-model value")


def construct_scale_chase(theta=None, base_grid=2001, coarse_grid=3, rounds=9, depth=3,
                          gamma=1.3):
    """Rigorous finite prefix of Corollary 2's constructed infinite sequence.

    M=c*M0 includes scaling its positive floor, so this family IS scale-closed.
    It is deliberately separate from the fixed-alpha production parametrization.
    Strict global negativity is certified on base_grid.  A lower interval bound
    proves rejection by the coarse ideal grid oracle.  After each rejection the
    raw learner receives epsilon_k=gamma*L_k*tau, rounded upward.  The next scale
    is a power of two at least twice the previous scale, selected by an exact
    rational comparison so its certified global slack exceeds epsilon_k.
    """
    if theta is None:
        # Analytic pullback-shaped member, selected independently of new outcomes.
        theta = np.array([[8., 0., 0.], [0., 16., 0.], [8., 0., 0.]])
    theta = np.asarray(theta, dtype=float)
    if not math.isfinite(gamma) or gamma <= 1 or rounds < 2:
        raise ValueError("coupled chase needs gamma>1 and at least two candidates")
    fine = cegis_c.verify(theta, base_grid, normalized=False, certified=True, cert_depth=depth)
    if not fine["certified"] or fine["uniform_slack"] <= 0 or fine["L"] <= 0:
        raise AssertionError("scale-chase base lacks strict global certification")
    model = bounds.assemble_exact(theta)
    points, nodes = cegis_c.grid(coarse_grid)
    from rounding import grid_radius
    tau = grid_radius(nodes, 2)
    intervals = bounds.contraction_grid_interval(theta, points, P=model)
    gap_lo = add_down(float(np.max(intervals.lo)), mul_down(fine["L"], tau))
    gap_hi = add_up(float(np.max(intervals.hi)), mul_up(fine["L"], tau))
    if gap_lo <= 0:
        raise AssertionError("coarse grid rejection not rigorously established")
    norm_L = bounds.L_norm(theta, depth=depth, P=model)
    base_normalized = bounds.contraction_grid_interval(theta, points, normalized=True, P=model)
    norm_gap = add_up(float(np.max(base_normalized.hi)), mul_up(norm_L, tau))
    b = bounds.b_upper_bound(theta, depth=depth, P=model)
    rows, exponent, applied_margin = [], 0, None
    eta0_exact = Fraction.from_float(float(fine["uniform_slack"]))
    for k in range(rounds):
        scale = math.ldexp(1.0, exponent)
        scaled = scale_polynomials(model, scale)
        # All coefficients must be exact c times the originals, including floor.
        for key in ("m11", "m12", "m22", "c11", "c12", "c22", "v"):
            if scaled[key].coeffs != {p: q * int(scale) for p, q in model[key].coeffs.items()}:
                raise AssertionError("coefficient scaling lost exactness")
        raw_values = bounds.contraction_grid_interval(theta, points, P=scaled)
        normalized_values = bounds.contraction_grid_interval(theta, points, normalized=True, P=scaled)
        L = math.ldexp(float(fine["L"]), exponent)  # exact power-of-two scaling
        if bounds.L_raw(theta, depth=depth, P=scaled) != L:
            raise AssertionError("raw coefficient oracle not scale-equivariant")
        eta = math.ldexp(float(fine["uniform_slack"]), exponent)
        next_margin = mul_up(gamma, mul_up(L, tau))
        raw_gap_lo = add_down(float(np.max(raw_values.lo)), mul_down(L, tau))
        raw_gap_hi = add_up(float(np.max(raw_values.hi)), mul_up(L, tau))
        normalized_gap_hi = add_up(float(np.max(normalized_values.hi)), mul_up(norm_L, tau))
        if raw_gap_lo <= 0 or (applied_margin is not None and not eta > applied_margin > 0):
            raise AssertionError("chase hypothesis or applied coupled margin failed")
        if not (np.array_equal(normalized_values.lo, base_normalized.lo) and
                np.array_equal(normalized_values.hi, base_normalized.hi)):
            raise AssertionError("normalized grid enclosure changed under exact scaling")
        rows.append({"round": k, "c": scale, "scale_exponent": exponent,
                     "raw_L_upper": L,
                     "raw_gap_lower": raw_gap_lo, "raw_gap_upper": raw_gap_hi,
                     "raw_training_margin": applied_margin,
                     "next_training_margin_upper": next_margin,
                     "raw_uniform_slack_lower": eta,
                     "metric_a_lower": scale * cegis_c.ALPHA,
                     "metric_b_upper": math.ldexp(b, exponent),
                     "normalized_L_upper": norm_L,
                     "normalized_gap_upper": normalized_gap_hi,
                     "zero_training_loss_for_any_dataset_in_domain": True if k else None,
                     "coarse_oracle_rejected_rigorously": True})
        # Use exact rationals for the selection inequality.  A fixed finite
        # numerical check is not substituted for the global slack certificate.
        exponent += 1
        while (Fraction(2) ** exponent) * eta0_exact <= Fraction.from_float(next_margin):
            exponent += 1
        applied_margin = next_margin
    return {"kind": "constructive existence illustration; not learner-frequency evidence",
            "family": "M_c=c*M0, c>=1, including scaled metric floor",
            "base_theta": theta.tolist(), "base_grid": base_grid,
            "coarse_grid": coarse_grid, "depth": depth, "tau_upper": tau,
            "gamma": gamma,
            "margin_rule": "epsilon_k=outward(gamma*L_k*tau), applied to candidate k+1",
            "scale_rule": "smallest power of two >=2*c_k with c_next*eta0_lower>epsilon_k; exact rational comparison",
            "domain": [-cegis_c.BOX, cegis_c.BOX], "rate": cegis_c.RATE,
            "base_uniform_slack_lower": fine["uniform_slack"],
            "base_raw_gap_interval": [gap_lo, gap_hi],
            "base_normalized_gap_upper": norm_gap,
            "checks": {"strict_global_negativity": True, "positive_L": True,
                       "strict_coarse_noncertification": True,
                       "exact_power_of_two_scaling": True,
                       "coupled_margin_satisfied_by_next_candidate": True,
                       "raw_bound_scale_equivariance": True,
                       "normalized_enclosure_invariant": True}, "rows": rows}


def run_pendulum_fixed(seed, pilot=False):
    """Matched fixed-margin control for the published pendulum protocol."""
    import bench_pendulum as pend
    rng = np.random.RandomState(seed)
    samples = 500 if not pilot else 80
    data = rng.uniform(-pend.BOX_P, pend.BOX_P, (samples, 2))
    theta = 0.1 * rng.randn(3, 3)
    theta[0, 0] = theta[2, 0] = 1.0
    grid, cap, steps = (401, 10, 500) if not pilot else (31, 2, 60)
    trace, timing, train_s, started = [], [], 0.0, time.perf_counter()
    for iteration in range(cap + 1):
        before = time.perf_counter()
        verified = pend.verify_p(theta, grid, normalized=True, depth=3)
        elapsed = time.perf_counter() - before
        trace.append({"iter": iteration, "L": verified["L"],
                      "tau": verified["tau"], "gap": verified["worst"] - verified["margin"],
                      "worst": verified["worst"], "uniform_slack": verified["uniform_slack"],
                      "grid_enclosed": verified["grid_enclosed"],
                      "outward_rounded": True, "eps": pend.BASE_MARGIN,
                      "parameter_norm": float(np.linalg.norm(theta)),
                      "theta": theta.tolist()})
        timing.append({"iter": iteration, "train_s": train_s,
                       "verify_including_bounds_s": elapsed,
                       **{k: verified[k] for k in ("bounds_s", "oracle_evaluations",
                           "nominal_grid_evaluations", "outward_grid_evaluations",
                           "gridpoints_covered")}})
        if verified["certified"] or iteration == cap:
            break
        indices = np.argsort(-verified["g"])[:pend.K_WORST]
        points = verified["X"][indices]
        perturbed = np.clip(np.repeat(points, pend.PERTURBATIONS, axis=0) +
                            pend.PERTURBATION_STD * rng.randn(pend.PERTURBATIONS * len(points), 2),
                            -pend.BOX_P, pend.BOX_P)
        data = np.vstack([data, points, perturbed])
        before = time.perf_counter()
        theta = pend._adam(theta, data, pend.BASE_MARGIN, steps=steps, lr=pend.LEARNING_RATE)
        train_s = time.perf_counter() - before
    verdict = experiments.classify(trace, bool(verified["certified"]))
    verdict.update(seed=seed, theta=theta.tolist(), elapsed_s=time.perf_counter() - started,
                   protocol={**pend.protocol_config(), "n_grid": grid, "coupled": False,
                             "max_rounds": cap, "training_steps": steps, "initial_samples": samples},
                   margin_units="lambda_max(C)/trace(M)", numerical_failure=False)
    return verdict, timing


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_results(data):
    """Check completed-study structure and positive-certificate invariants.

    This validates recorded evidence rather than rerunning the numerical oracle.
    The integrated pipeline still executes the oracle and focused arithmetic tests.
    """
    if data.get("schema_version") != 1 or data.get("status") != "pass":
        raise ValueError("revision study is incomplete or has an unknown schema")
    pilot = data.get("pilot")
    if not isinstance(pilot, bool):
        raise ValueError("missing pilot/production distinction")
    expected_seeds = list(range(2 if pilot else 20))
    if data["config"]["seeds"] != expected_seeds:
        raise ValueError("unexpected controlled seed matrix")
    expected_names = {"raw_fixed", "raw_coupled", "normalized_fixed", "normalized_coupled",
                      "raw_coupled_bounded", "normalized_coupled_bounded",
                      "normalized_coupled_bounded_coarse"}
    if set(data["regimes"]) != expected_names or set(data["summary"]) != expected_names:
        raise ValueError("missing controlled experiment regime")
    production_grid = 31 if pilot else 2001

    def validate_runs(runs, summary, seeds, bounded=False):
        if set(runs) != set(map(str, seeds)):
            raise ValueError("incomplete seed matrix")
        if summary != summarize(runs):
            raise ValueError("summary does not match saved individual outcomes")
        if sum(summary[k] for k in ("certified", "diverged", "stalled", "numerical_failure")) != len(seeds):
            raise ValueError("outcome counts do not partition the seed set")
        for seed, run in runs.items():
            if run["seed"] != int(seed):
                raise ValueError("seed identity mismatch")
            if sum(bool(run.get(k, False)) for k in
                   ("certified", "diverged", "stalled", "numerical_failure")) != 1:
                raise ValueError("individual outcome labels do not partition")
            trace = run["trace"]
            if not trace or [row["iter"] for row in trace] != list(range(len(trace))):
                raise ValueError("nonsequential verification trace")
            if bounded:
                radius = run["protocol"]["radius"]
                tolerance = 8 * np.finfo(float).eps * max(1., radius)
                if radius != data["config"]["radius"]:
                    raise ValueError("bounded radius changed from predeclared rule")
                for row in trace:
                    radius2 = Fraction.from_float(float(radius)) ** 2
                    if (not row["exact_parameter_ball_enforced"] or
                        Fraction(row["max_parameter_norm_squared_exact"]) > radius2 or
                        exact_norm_squared(row["theta"]) > radius2):
                        raise ValueError("an Adam update escaped the exact parameter ball")
                    if max(row["parameter_norm"], row["max_post_projection_norm"]) > radius + tolerance:
                        raise ValueError("an Adam update escaped the parameter ball")
            if run["certified"]:
                final = trace[-1]
                if not final["grid_enclosed"] or not final["outward_rounded"] or final["uniform_slack"] < 0:
                    raise ValueError("certified run lacks directed-rounding evidence")
                if run["iters_to_cert"] != len(trace) - 1:
                    raise ValueError("incorrect certification iteration")
                if final.get("grid_values_kind", "outward_upper") != "outward_upper":
                    raise ValueError("positive certificate used screening values")
                if "raw_margin_lower" in run:
                    slack = final["uniform_slack"]
                    expected = max(0.0, mul_down(2.0 * cegis_c.ALPHA, slack)) if run["normalized"] else slack
                    if run["raw_margin_lower"] != expected or run["raw_margin_lower"] < 0:
                        raise ValueError("raw margin does not match saved directed slack")
                    if not run["metric_bounds_outward_rounded"] or run["b_upper"] < run["a"]:
                        raise ValueError("invalid metric class bounds")
            for row in trace:
                if not all(math.isfinite(float(row[key])) for key in
                           ("L", "tau", "gap", "uniform_slack", "parameter_norm")):
                    raise ValueError("nonfinite trace quantity")

    for name, runs in data["regimes"].items():
        validate_runs(runs, data["summary"][name], expected_seeds, "bounded" in name)
        for run in runs.values():
            protocol = run["protocol"]
            expected_grid = (31 if pilot else 121) if name.endswith("coarse") else production_grid
            if protocol["n_grid"] != expected_grid:
                raise ValueError("uncontrolled grid difference")
            if not name.endswith("coarse"):
                expected = data["config"]["factorial_protocol"]
                for key in expected:
                    if key != "radius" and protocol[key] != expected[key]:
                        raise ValueError(f"uncontrolled protocol difference in {key}")
    for seed in map(str, expected_seeds):
        paired = [runs[seed] for name, runs in data["regimes"].items()
                  if not name.endswith("coarse")]
        for key in ("initial_dataset_sha256", "initial_theta_sha256"):
            if len({run[key] for run in paired}) != 1:
                raise ValueError("paired regimes did not share their initialization")
    chase = data["scale_chase"]
    if not chase["checks"] or not all(chase["checks"].values()):
        raise ValueError("constructive scale-chase hypothesis failed")
    if chase["base_uniform_slack_lower"] <= 0 or chase["base_raw_gap_interval"][0] <= 0:
        raise ValueError("chase base is not strictly valid and uncertified on the coarse grid")
    initial = chase["rows"][0]
    for k, row in enumerate(chase["rows"]):
        if row["round"] != k or row["c"] != math.ldexp(1., row["scale_exponent"]):
            raise ValueError("chase scale is not the prescribed power of two")
        if not row["coarse_oracle_rejected_rigorously"] or row["raw_gap_lower"] <= 0:
            raise ValueError("chase lost coarse noncertification")
        if row["next_training_margin_upper"] != mul_up(
                chase["gamma"], mul_up(row["raw_L_upper"], chase["tau_upper"])):
            raise ValueError("chase training margin is not coupled to its previous oracle")
        if k == 0:
            if row["raw_training_margin"] is not None or row["c"] != 1.:
                raise ValueError("chase base incorrectly has an applied training margin")
        else:
            previous = chase["rows"][k-1]
            if row["raw_training_margin"] != previous["next_training_margin_upper"]:
                raise ValueError("chase applied the wrong preceding coupled margin")
            if not row["raw_uniform_slack_lower"] > row["raw_training_margin"] > 0:
                raise ValueError("constructed candidate fails its coupled training margin")
            if (row["c"] < 2 * previous["c"] or
                Fraction.from_float(row["c"]) * Fraction.from_float(chase["base_uniform_slack_lower"])
                    <= Fraction.from_float(row["raw_training_margin"])):
                raise ValueError("chase exact global-slack scaling inequality failed")
        if row["raw_L_upper"] != math.ldexp(initial["raw_L_upper"], row["scale_exponent"]):
            raise ValueError("raw chase bound did not scale exactly")
        if row["normalized_gap_upper"] != initial["normalized_gap_upper"]:
            raise ValueError("normalized chase gap changed")
    sensitivity = data["sensitivity"]
    sens_seeds = [0] if pilot else list(range(5))
    if sensitivity["seeds"] != sens_seeds or len(sensitivity["regimes"]) != 5:
        raise ValueError("incomplete predeclared sensitivity matrix")
    for name, runs in sensitivity["regimes"].items():
        validate_runs(runs, sensitivity["summary"][name], sens_seeds)
        for run in runs.values():
            if run["protocol"]["n_grid"] != 31:
                raise ValueError("sensitivity changed its common coarse grid")
    pendulum = data["pendulum_fixed"]
    validate_runs(pendulum["runs"], pendulum["summary"], expected_seeds)
    return True


def plot_ablation(result, destination):
    names, summaries = list(result["summary"]), result["summary"]
    # A single full-column chart keeps seven regime labels readable in print.
    # Per-regime and per-round costs remain in the generated timing tables.
    fig, axis = plt.subplots(figsize=(6.8, 3.6), constrained_layout=True)
    positions, base = np.arange(len(names)), np.zeros(len(names))
    for key, color in (("certified", "tab:green"), ("stalled", "tab:orange"),
                       ("diverged", "tab:red"), ("numerical_failure", "tab:gray")):
        vals = np.array([summaries[name][key] for name in names])
        axis.bar(positions, vals, bottom=base, label=key.replace("_", " "), color=color)
        base += vals
    axis.set_ylabel("Number of seeds")
    axis.legend(fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    from matplotlib.ticker import MaxNLocator
    axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    labels = [name.replace("normalized", "norm.").replace("_", "\n") for name in names]
    axis.set_xticks(positions, labels, fontsize=9)
    fig.savefig(destination)
    plt.close(fig)


def plot_chase(chase, destination):
    rows = chase["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    c = [r["c"] for r in rows]
    axes[0].loglog(c, [r["raw_gap_lower"] for r in rows], "o-", label="raw gap lower bound")
    axes[0].loglog(c[1:], [r["raw_training_margin"] for r in rows[1:]], "s-",
                   label="applied coupled raw margin")
    axes[0].set_ylabel("Raw quantities")
    axes[0].legend(fontsize=8)
    axes[1].semilogx(c, [r["normalized_gap_upper"] for r in rows], "o-", color="tab:purple")
    axes[1].set_ylabel("Normalized gap upper bound (invariant)")
    for axis in axes:
        axis.set_xlabel("Exact metric scale c")
    fig.savefig(destination)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv)
    suffix = "_pilot" if args.pilot else ""
    results_path = HERE / "results" / f"revision_experiments{suffix}.json"
    timing_path = HERE / "results" / f"timing_revision{suffix}.json"
    seeds = list(range(2 if args.pilot else 20))
    common = Protocol() if not args.pilot else Protocol(n_grid=31, max_rounds=2,
                                                       training_steps=60, initial_samples=80)
    radius = radius_rule()
    regimes = [(f"{'normalized' if normalized else 'raw'}_{'coupled' if coupled else 'fixed'}",
                normalized, coupled, common)
               for normalized in (False, True) for coupled in (False, True)]
    regimes += [(f"{'normalized' if normalized else 'raw'}_coupled_bounded", normalized, True,
                 replace(common, radius=radius)) for normalized in (False, True)]
    regimes += [("normalized_coupled_bounded_coarse", True, True,
                 replace(common, n_grid=31 if args.pilot else 121, radius=radius))]
    result = {"schema_version": 1, "status": "running", "pilot": args.pilot,
              "verification_backend": backend_info(),
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "machine": platform.machine(), "processor": platform.processor(),
                              "thread_environment": {key: os.environ.get(key) for key in
                                  ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                                   "VECLIB_MAXIMUM_THREADS")}},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "config": {"seeds": seeds, "domain": [-cegis_c.BOX, cegis_c.BOX],
                         "rate": cegis_c.RATE, "metric_floor": cegis_c.ALPHA,
                         "factorial_protocol": asdict(common),
                         "radius": radius,
                         "radius_selection": "4*Frobenius norm of common identity initialization; no outcome tuning",
                         "bounded_learner": "projected Adam at every update; exact rational norm checks on stored parameters; finite-budget heuristic, not exact feasibility",
                         "margin_comparison": "same numeric base margin 0.05 in each selected test's units; not matched raw/normalized physical slack",
                         "randomization": "per-seed RandomState reset, common initial samples and perturbation draws for equal k; data locations can diverge",
                         "outcome_criterion": experiments.protocol_config()["outcome_criterion"]},
              "regimes": {}, "summary": {}, "sensitivity": {}, "pendulum_fixed": {}}
    timing = {"units": "seconds", "note": "train_s precedes verification; bound computation included in verification", "runs": {}}
    started = time.perf_counter()
    for name, normalized, coupled, config in regimes:
        result["regimes"][name], timing["runs"][name] = {}, {}
        for seed in seeds:
            run, costs = run_controlled(seed, normalized, coupled, config)
            result["regimes"][name][str(seed)] = run
            timing["runs"][name][str(seed)] = costs
            print(f"{name} seed={seed}: certified={run['certified']} rounds={len(run['trace'])-1} elapsed={run['elapsed_s']:.1f}s", flush=True)
            result["summary"][name] = summarize(result["regimes"][name])
            write_json(results_path, result)
            write_json(timing_path, timing)
    print("Constructing rigorously checked scale-chase illustration", flush=True)
    result["scale_chase"] = construct_scale_chase()
    sensitivity_config = replace(common, n_grid=31, max_rounds=2 if args.pilot else 4,
                                 training_steps=60 if args.pilot else 500)
    sensitivity_seeds = [0] if args.pilot else list(range(5))
    result["sensitivity"] = {"kind": "short-horizon coarse-grid diagnostic; not operating-grid success rates",
                              "config": asdict(sensitivity_config), "seeds": sensitivity_seeds,
                              "regimes": {}, "summary": {}}
    for gamma, worst in ((1.1, 60), (1.3, 60), (2.0, 60), (1.3, 10), (1.3, "all")):
        name = f"sensitivity_gamma_{gamma}_k_{worst}"
        runs, timing["runs"][name] = {}, {}
        for seed in sensitivity_seeds:
            run, costs = run_controlled(seed, True, True,
                                        replace(sensitivity_config, gamma=gamma, k_worst=worst))
            runs[str(seed)], timing["runs"][name][str(seed)] = run, costs
            print(f"{name} seed={seed}: certified={run['certified']} elapsed={run['elapsed_s']:.1f}s", flush=True)
        result["sensitivity"]["regimes"][name] = runs
        result["sensitivity"]["summary"][name] = summarize(runs)
        write_json(results_path, result)
        write_json(timing_path, timing)
    pendulum_runs, timing["runs"]["pendulum_fixed"] = {}, {}
    for seed in seeds:
        run, costs = run_pendulum_fixed(seed, pilot=args.pilot)
        pendulum_runs[str(seed)], timing["runs"]["pendulum_fixed"][str(seed)] = run, costs
        print(f"pendulum_fixed seed={seed}: certified={run['certified']} elapsed={run['elapsed_s']:.1f}s", flush=True)
    result["pendulum_fixed"] = {"runs": pendulum_runs, "summary": summarize(pendulum_runs)}
    result.update(status="pass", elapsed_s=time.perf_counter() - started)
    validate_results(result)
    paper = HERE / "paper"
    paper.mkdir(exist_ok=True)
    plot_ablation(result, paper / f"fig_ablation{suffix}.pdf")
    plot_chase(result["scale_chase"], paper / f"fig_scale_chase{suffix}.pdf")
    write_json(results_path, result)
    write_json(timing_path, timing)
    print(f"Wrote {results_path}; elapsed={result['elapsed_s']:.1f}s", flush=True)
    return result


if __name__ == "__main__":
    main()
