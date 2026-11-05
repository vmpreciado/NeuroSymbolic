"""experiments.py: multi-seed study of the CEGIS-C loop with the CERTIFIED
Lipschitz oracle. Writes results/traces.json (per-seed traces + per-regime
summary) and results/gridcost.json (slack/L ratio across certified seeds).
Regenerates fig_traces.pdf, fig_heatmaps.pdf, fig_gridcost.pdf.

All three regimes run the normalized test g_tilde, so their certifiability
gaps live on one scale-invariant axis (a common axis, per the referee). They
differ only in the margin rule and the grid resolution:

  stall    fixed margin eps0, coarse grid 121. eps0 < L_h*tau, so the stall
           band can be nonempty; noncertifying bounded-growth runs are labeled stalled.
  diverge  coupled margin gamma*L_h*tau, coarse grid 121. The required margin
           outgrows the slack achieved in these runs; uncertified runs whose
           Lipschitz bound crosses the recorded growth criterion are labeled
           diverged. This shape-dependent outcome differs from raw-test scale chasing.
  certify  coupled margin, fine grid 2001. Here gamma*L_h*tau falls below the
           achieved slack in successful runs, and Algorithm 1 certifies.

The operating grid for certification is finer than the prototype's 121 because
the certified oracle is more conservative; the feasibility condition is
tau < slack / (gamma * L_h), which ties directly to the grid-cost analysis.
"""
import json
import os
import time
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cegis_c
import bounds
from rounding import backend_info
from cegis_c import lam_max_C, lam_max_C_norm, grid, BOX, EPS, adam, perturb

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PAPER = os.path.join(HERE, "paper")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(PAPER, exist_ok=True)


SEEDS = list(range(20))
GRID_COARSE = 121       # stall and diverge regimes
GRID_OPERATING = 2001   # certification regime (feasible coupled margin)
MAX_ITER = 12
DEPTH = 3
GROWTH_FACTOR = 100.0
TRAINING_STEPS = 500
LEARNING_RATE = 2e-2
INITIAL_SAMPLES = 500
GAMMA = 1.3
K_WORST = 60
PERTURBATIONS = 5
PERTURBATION_STD = 0.08

REGIMES = [
    ("stall",   dict(couple_margin=False, normalized=True, n_grid=GRID_COARSE)),
    ("diverge", dict(couple_margin=True,  normalized=True, n_grid=GRID_COARSE)),
    ("certify", dict(couple_margin=True,  normalized=True, n_grid=GRID_OPERATING)),
]


def identity_init():
    th = np.zeros((3, 3))
    th[0, 0] = 1.0
    th[2, 0] = 1.0
    return th


def run_variant(couple_margin, normalized, n_grid=GRID_OPERATING, max_iter=MAX_ITER,
                k_worst=K_WORST, m_pert=PERTURBATIONS, seed=0, timings=None, depth=DEPTH):
    """One CEGIS-C run. Verify-first; per-iteration trace records the
    certifiability gap = worst test value - oracle's required margin."""
    npr.seed(seed)
    D = npr.uniform(-BOX, BOX, size=(INITIAL_SAMPLES, 2))
    theta = identity_init()
    eps = EPS
    trace = []
    train_s = 0.0
    for it in range(max_iter + 1):
        t_verify = time.perf_counter()
        V = cegis_c.verify(theta, n_grid, normalized=normalized,
                           certified=True, cert_depth=depth)
        verify_s = time.perf_counter() - t_verify
        outward_evaluations = n_grid ** 2 if V["grid_enclosed"] else 0
        oracle_evaluations = n_grid ** 2 + outward_evaluations
        if timings is not None:
            timings.append({"iter": it, "train_s": train_s,
                            "verify_including_bounds_s": verify_s,
                            "oracle_evaluations": oracle_evaluations,
                            "nominal_grid_evaluations": n_grid ** 2,
                            "outward_grid_evaluations": outward_evaluations})
        trace.append({"iter": it, "worst": V["worst"], "margin": V["margin"],
                      "gap": V["worst"] - V["margin"], "viol": int(V["viol"].sum()),
                      "eps": eps, "L": V["L"], "tau": V["tau"],
                      "grid_enclosed": V["grid_enclosed"],
                      "outward_rounded": V["outward_rounded"],
                      "uniform_slack": V["uniform_slack"],
                      "grid_values_kind": V["grid_values_kind"],
                      "oracle_evaluations": oracle_evaluations,
                      "nominal_grid_evaluations": n_grid ** 2,
                      "outward_grid_evaluations": outward_evaluations})
        if V["certified"] or it == max_iter:
            return theta, trace, V, bool(V["certified"])
        if couple_margin:
            eps = max(EPS, GAMMA * V["L"] * V["tau"])
        idx = np.argsort(-V["g"])[:k_worst]
        Xc = V["X"][idx]
        D = np.vstack([D, Xc, perturb(Xc, m_pert, sigma=PERTURBATION_STD)])
        t_train = time.perf_counter()
        theta = adam(theta, D, eps, steps=TRAINING_STEPS, lr=LEARNING_RATE,
                     normalized=normalized)
        train_s = time.perf_counter() - t_train
    return theta, trace, V, bool(V["certified"])


def classify(trace, certified):
    """Per-seed verdict. The three outcomes partition the runs: a run either
    certifies, or diverges (uncertified with the certified Lipschitz constant
    inflating past 100x its initial value), or stalls (uncertified with a
    bounded constant within the iteration budget). This outcome label does not
    establish the global loss and stationarity hypotheses of the stall lemma."""
    Ls = [t["L"] for t in trace]
    gaps = [t["gap"] for t in trace]
    L0, Lmax = Ls[0], max(Ls)
    diverged = (not certified) and (Lmax > GROWTH_FACTOR * L0)
    stalled = (not certified) and (not diverged)
    return {
        "certified": bool(certified),
        "diverged": bool(diverged),
        "stalled": bool(stalled),
        "iters_to_cert": (len(trace) - 1) if certified else None,
        "max_L": float(Lmax),
        "initial_L": float(L0),
        "L_growth_factor": float(Lmax / L0) if L0 > 0 else float("nan"),
        "final_gap": float(gaps[-1]),
        "trace": trace,
    }


def protocol_config():
    """The settings actually used by the production multi-seed experiment."""
    return {
        "max_rounds": MAX_ITER, "growth_factor": GROWTH_FACTOR,
        "training_steps": TRAINING_STEPS, "learning_rate": LEARNING_RATE,
        "initial_samples": INITIAL_SAMPLES, "base_margin": EPS, "gamma": GAMMA,
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


def slack_and_L_certified(theta):
    """Sampled normalized slack, certified L, and certified upper bound b.
    Slack is minus the largest gt value on a diagnostic 401-point-per-axis
    grid, not a certified uniform slack. L bounds the Lipschitz constant;
    b is the certified
    bound on lambda_max(M) from Lemma B(iii) (a = ALPHA exactly)."""
    X, _ = grid(401)
    gt = lam_max_C_norm(theta, X)
    slack = -float(np.max(gt))
    L = float(bounds.L_norm(theta, depth=DEPTH))
    b = float(bounds.b_upper_bound(theta, depth=DEPTH))
    return slack, L, b


def main():
    print(f"=== multi-seed regimes ({len(SEEDS)} seeds, depth={DEPTH}) ===")
    all_runs = {name: {} for name, _ in REGIMES}
    final_thetas = {name: {} for name, _ in REGIMES}
    timings = {name: {} for name, _ in REGIMES}

    t0 = time.time()
    for name, kw in REGIMES:
        for seed in SEEDS:
            t_seed = time.time()
            timing = []
            theta, trace, V, ok = run_variant(seed=seed, max_iter=MAX_ITER,
                                             timings=timing, **kw)
            timings[name][str(seed)] = timing
            verdict = classify(trace, ok)
            verdict["theta"] = theta.tolist()
            all_runs[name][str(seed)] = verdict
            final_thetas[name][seed] = theta
            dt = time.time() - t_seed
            print(f"  {name:8} seed {seed:2d}: cert={ok} iters={len(trace)-1:2d} "
                  f"L0={verdict['initial_L']:.2f} Lmax={verdict['max_L']:.2e} "
                  f"gap={verdict['final_gap']:+.4f}  ({dt:.1f}s)")
    print(f"total runtime: {time.time() - t0:.1f}s")

    # ---------------- summary stats ----------------
    summary = {}
    for name in all_runs:
        seeds = list(all_runs[name].values())
        cert = sum(1 for r in seeds if r["certified"])
        div = sum(1 for r in seeds if r["diverged"])
        sta = sum(1 for r in seeds if r["stalled"])
        iters = [r["iters_to_cert"] for r in seeds if r["iters_to_cert"] is not None]
        Lgrowths = [r["L_growth_factor"] for r in seeds]
        summary[name] = {
            "n": len(seeds),
            "certified": cert,
            "diverged": div,
            "stalled": sta,
            "median_iters_to_cert": float(np.median(iters)) if iters else None,
            "iqr_iters_to_cert": [float(np.percentile(iters, 25)),
                                  float(np.percentile(iters, 75))] if iters else None,
            "max_L_growth_factor": float(max(Lgrowths)) if Lgrowths else None,
        }
    print()
    for name, s in summary.items():
        print(f"  {name:8}: certified {s['certified']}/{s['n']}, "
              f"diverged {s['diverged']}, stalled {s['stalled']}, "
              f"median iters {s['median_iters_to_cert']}, "
              f"max L growth {s['max_L_growth_factor']:.2e}")

    out_traces = {
        "verification_backend": backend_info(),
        "config": {"n_seeds": len(SEEDS), "grid_coarse": GRID_COARSE,
                   "grid_operating": GRID_OPERATING, "max_iter": MAX_ITER,
                   "depth": DEPTH, "BOX": BOX, "ALPHA": float(cegis_c.ALPHA),
                   "RATE": float(cegis_c.RATE), "eps0": EPS, **protocol_config()},
        "regimes": all_runs,
        "summary": summary,
    }
    with open(os.path.join(RESULTS, "traces.json"), "w") as f:
        json.dump(out_traces, f, indent=2, default=float)
    print(f"  wrote {os.path.join(RESULTS, 'traces.json')}")
    with open(os.path.join(RESULTS, "timing_polynomial.json"), "w") as f:
        json.dump({"units": "seconds", "note": "train_s precedes the recorded verification; verify includes bound construction", "runs": timings}, f, indent=2)

    # ---------- gridcost: slack/L ratio across certified 'certify'-regime seeds ----------
    print("\n=== gridcost: slack/L ratio across certified seeds ===")
    ratios = []
    cert_seeds = [s for s in SEEDS if all_runs["certify"][str(s)]["certified"]]
    for seed in cert_seeds:
        slack, L, b = slack_and_L_certified(final_thetas["certify"][seed])
        if L > 0 and slack > 0:
            ratios.append((seed, slack, L, slack / L, b))
            print(f"  seed {seed:2d}: slack={slack:.4f}  L_cert={L:.2f}  "
                  f"b={b:.3f}  slack/L={slack / L:.5f}")
    rs = [r[3] for r in ratios]
    median_ratio = float(np.median(rs)) if rs else None
    min_ratio = float(min(rs)) if rs else None
    max_ratio = float(max(rs)) if rs else None
    bs = [r[4] for r in ratios]
    median_b = float(np.median(bs)) if bs else None
    print(f"  ratio: median={median_ratio} min={min_ratio} max={max_ratio}")

    dims = list(range(2, 9))
    if median_ratio:
        pts = [((2 * BOX) / (2 * median_ratio / np.sqrt(d)) + 1) ** d for d in dims]
    else:
        pts = [None] * len(dims)
    out_gridcost = {
        "verification_backend": backend_info(),
        "config": {"grid_operating": GRID_OPERATING, "depth": DEPTH,
                   "n_seeds": len(SEEDS), "n_certified": len(cert_seeds)},
        "interpretation": "Higher-dimensional counts extrapolate the measured 2D grid slack/L ratio; they are not higher-dimensional experiments or certified uniform slack estimates.",
        "a": float(cegis_c.ALPHA),
        "median_b": median_b,
        "ratios": [{"seed": r[0], "slack": r[1], "L": r[2], "ratio": r[3],
                    "slack_kind": "sampled_on_401_by_401_grid",
                    "b": r[4], "a": float(cegis_c.ALPHA)}
                   for r in ratios],
        "median_ratio": median_ratio,
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
        "dims": dims,
        "grid_points": pts,
    }
    with open(os.path.join(RESULTS, "gridcost.json"), "w") as f:
        json.dump(out_gridcost, f, indent=2, default=float)
    print(f"  wrote {os.path.join(RESULTS, 'gridcost.json')}")

    # ---------------- figures ----------------
    print("\n=== regenerate fig_heatmaps, fig_traces, fig_gridcost ===")
    th_repr = final_thetas["certify"][cert_seeds[0]] if cert_seeds else identity_init()
    plot_heatmaps(th_repr)
    plot_traces(all_runs)
    if median_ratio:
        plot_gridcost(median_ratio, dims, pts)
    print("  figures written")


# ---------------- plots ----------------

def plot_heatmaps(th_cert):
    n = 301
    X, s = grid(n)
    g0 = lam_max_C(identity_init(), X).reshape(n, n)
    g1 = lam_max_C(th_cert, X).reshape(n, n)
    vmax = float(np.max(np.abs(g0)))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for ax, G, title in [(axes[0], g0, "identity metric (initial candidate)"),
                         (axes[1], g1, "certified metric (CEGIS-C output)")]:
        im = ax.imshow(G, origin="lower", extent=[-BOX, BOX, -BOX, BOX],
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.contour(s, s, G, levels=[0.0], colors="k", linewidths=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    cb = fig.colorbar(im, ax=axes, shrink=0.9)
    cb.set_label(r"$\lambda_{\max}(C_\theta(x))$")
    plt.savefig(os.path.join(PAPER, "fig_heatmaps.pdf"))
    plt.savefig(os.path.join(PAPER, "fig_heatmaps.png"), dpi=160)
    plt.close()


def plot_traces(all_runs):
    """Normalized certifiability gap per iteration for the three regimes, all
    on the scale-invariant normalized test, so the axis is common. A symlog
    y-axis shows both the positive (uncertified) and negative (certified)
    gap, so certification is visible as a crossing below zero. Thin lines per
    seed, bold median per regime, legend reports the outcome counts."""
    fig, ax = plt.subplots(figsize=(6.0, 3.7), constrained_layout=True)
    style = {
        "stall":   ("tab:red",    "fixed margin, coarse grid"),
        "diverge": ("tab:orange", "coupled margin, coarse grid"),
        "certify": ("tab:blue",   "coupled margin, operating grid"),
    }
    for name in ["stall", "diverge", "certify"]:
        color, base = style[name]
        runs = all_runs[name]
        n = len(runs)
        cert = sum(1 for r in runs.values() if r["certified"])
        div = sum(1 for r in runs.values() if r["diverged"])
        sta = sum(1 for r in runs.values() if r["stalled"])
        all_gaps, max_len = [], 0
        for verdict in runs.values():
            gaps = [t["gap"] for t in verdict["trace"]]
            its = [t["iter"] for t in verdict["trace"]]
            ax.plot(its, gaps, color=color, alpha=0.16, lw=0.8)
            all_gaps.append(gaps)
            max_len = max(max_len, len(gaps))
        padded = np.array([g + [g[-1]] * (max_len - len(g)) for g in all_gaps])
        med = np.median(padded, axis=0)
        counts = {"certify": f"{cert}/{n} certified",
                  "diverge": f"{div}/{n} diverged",
                  "stall": f"{sta}/{n} stalled"}[name]
        ax.plot(range(max_len), med, color=color, lw=2.4, label=f"{base}, {counts}")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.axhline(0.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("CEGIS iteration")
    ax.set_ylabel(r"normalized certifiability gap $\Gamma$ (certified $\Leftrightarrow \Gamma \leq 0$)")
    ax.legend(fontsize=7.5, loc="upper right")
    plt.savefig(os.path.join(PAPER, "fig_traces.pdf"))
    plt.savefig(os.path.join(PAPER, "fig_traces.png"), dpi=160)
    plt.close()


def plot_gridcost(median_ratio, dims, pts):
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    ax.semilogy(dims, pts, "o-", color="tab:purple")
    ax.axhline(1e9, color="gray", ls=":", lw=1)
    ax.text(2.1, 2e9, "$10^9$ grid points", fontsize=8, color="gray")
    ax.set_xlabel(r"state dimension $n$")
    ax.set_ylabel("extrapolated grid points")
    ax.set_title(f"grid cost estimate (median sampled slack/$L_{{\\mathrm{{cert}}}}$"
                 f"$={median_ratio:.4f}$)", fontsize=10)
    plt.savefig(os.path.join(PAPER, "fig_gridcost.pdf"))
    plt.savefig(os.path.join(PAPER, "fig_gridcost.png"), dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
