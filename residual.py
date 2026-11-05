"""residual.py: Theorem 2 (residual-aware certificate) experiment.

Given a certified metric M_theta with uniform margin eta on X, perturb the
dynamics to f + d with the polynomial residual
    d(x) = delta * 0.25 * (x_2^2,  x_1 * x_2),
sweep delta, and compare
  - the measured worst value of lambda_max(C_true(x)) on the verification
    grid, where C_true is the true contraction matrix for f + d in M_theta;
  - the theoretical bound -eta + mu_1 * delta_0(delta) + 2 b * delta_1(delta)
    from Theorem 2.

Outputs results/residual.json and paper/fig_residual.pdf.
"""
import json
import math
import os
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cegis_c
import bounds
from bounds import Poly2, ExactPoly2
from rounding import (add_up, mul_up, sub_down, div_down, sqrt_up, backend_info)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PAPER = os.path.join(HERE, "paper")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(PAPER, exist_ok=True)


def select_certified_theta(n_grid_train=2001, max_iter=12, seed=0):
    """Train the normalized-coupled variant with the certified oracle at the
    operating grid (2001), where the coupled margin gamma*L_h*tau is feasible
    against the achievable slack and the loop certifies reliably across seeds
    (see results/traces.json). Returns the certified candidate metric."""
    import experiments
    th, tr, V, ok = experiments.run_variant(True, True, n_grid=n_grid_train,
                                            max_iter=max_iter, seed=seed)
    return th, V, ok


def assemble_true(theta, delta):
    """Build the entries of C_true(x) for the perturbed dynamics f + d(x),
    in the metric M_theta, as Poly2 polynomials.

    C_true(x) = C(x) + sum_k (dM/dx_k)(x) * d_k(x) + D(x)^T M(x) + M(x) D(x),
    where D(x) = partial d / partial x.

    d(x) = delta * 0.25 * (x_2^2, x_1 * x_2).
    D(x) = delta * [[0, 0.5 x_2], [0.25 x_2, 0.25 x_1]]
    """
    P = bounds.assemble(theta)
    # d entries
    d1 = Poly2.x2() * Poly2.x2() * (delta * 0.25)        # delta * 0.25 * x2^2
    d2 = Poly2.x1() * Poly2.x2() * (delta * 0.25)        # delta * 0.25 * x1 x2
    d = [d1, d2]
    # D = partial d / partial x  (rows: d1; d2.  columns: x1, x2)
    D11 = d1.deriv(0); D12 = d1.deriv(1)
    D21 = d2.deriv(0); D22 = d2.deriv(1)
    # extra term sum_k (dM/dxk) * d_k  (entrywise, symmetric)
    e11 = P["dm"][0][0] * d[0] + P["dm"][1][0] * d[1]
    e12 = P["dm"][0][1] * d[0] + P["dm"][1][1] * d[1]
    e22 = P["dm"][0][2] * d[0] + P["dm"][1][2] * d[1]
    # D^T M + M D
    m11, m12, m22 = P["m11"], P["m12"], P["m22"]
    # entry-wise of symmetric matrix S = D^T M + M D
    # S = D^T M + M D, where D is 2x2.  S is symmetric.
    s11 = D11 * m11 * 2.0 + D21 * m12 * 2.0
    s22 = D22 * m22 * 2.0 + D12 * m12 * 2.0
    s12 = D11 * m12 + D21 * m22 + D12 * m11 + D22 * m12
    c11_t = P["c11"] + e11 + s11
    c12_t = P["c12"] + e12 + s12
    c22_t = P["c22"] + e22 + s22
    return c11_t, c12_t, c22_t


def lam_max_2x2(c11, c12, c22):
    """Vectorized lambda_max of symmetric 2x2 [[c11, c12], [c12, c22]]."""
    m = 0.5 * (c11 + c22)
    d = 0.5 * (c11 - c22)
    return m + np.sqrt(d ** 2 + c12 ** 2 + 1e-300)


def residual_certified_constants(theta, depth=3, n_grid_eta=4001):
    """Compute eta, mu1, b_upper, c_d0, c_d1, delta_star for the given theta."""
    P = bounds.assemble_exact(theta)
    # Force a full outward grid check: a nominal sampled maximum is insufficient.
    enclosure = cegis_c.grid_enclosure(theta, n_grid_eta, normalized=False, P=P)
    worst_raw = enclosure["worst_upper"]
    Lg = bounds.L_raw(theta, depth=depth, P=P)
    tau = enclosure["tau"]
    eta = sub_down(-worst_raw, mul_up(Lg, tau))
    # b_upper and mu1 from bounds.py
    b_upper = bounds.b_upper_bound(theta, depth=depth, P=P)
    mu1 = bounds.mu1_upper_bound(theta, depth=depth, P=P)
    # certified constants for the residual family d(x) = delta * 0.25 * (x2^2, x1 x2)
    # ||d_unit(x)||^2 = (0.25)^2 * (x2^4 + x1^2 x2^2)
    d_unit_norm2 = ExactPoly2({(0, 4): 1.0, (2, 2): 1.0}) * 0.0625
    c_d0 = sqrt_up(d_unit_norm2.supabs_subdiv(2.0, depth))
    # ||J_d_unit||_F^2 = (0.5 x2)^2 + (0.25 x2)^2 + (0.25 x1)^2
    #                  = 0.25 x2^2 + 0.0625 x2^2 + 0.0625 x1^2
    #                  = 0.3125 x2^2 + 0.0625 x1^2
    Jd_unit_F2 = ExactPoly2({(0, 2): 0.3125, (2, 0): 0.0625})
    c_d1 = sqrt_up(Jd_unit_F2.supabs_subdiv(2.0, depth))
    # delta_star: eta == mu1*c_d0*delta + 2 b_upper*c_d1*delta
    denom = add_up(mul_up(mu1, c_d0), mul_up(mul_up(2.0, b_upper), c_d1))
    if denom <= 0:
        raise ValueError("invalid residual sensitivity denominator")
    delta_star = div_down(eta, denom) if eta > 0 else 0.0
    return {
        "Lg": Lg, "tau": tau, "worst_raw_grid": worst_raw,
        "eta": eta, "b_upper": b_upper, "mu1": mu1,
        "c_d0": c_d0, "c_d1": c_d1,
        "delta_star": delta_star,
        "n_grid_eta": n_grid_eta,
        "grid_enclosed": bool(enclosure["grid_enclosed"]), "outward_rounded": bool(enclosure["outward_rounded"]),
        "rounding_mode": "outward", "budget_outward_rounded": bool(eta > 0),
    }


def sweep(theta, info, deltas, n_grid_meas=2001):
    """For each delta, measure worst lambda_max(C_true) on the grid; compare
    with the theoretical bound."""
    X, _ = cegis_c.grid(n_grid_meas)
    rows = []
    for delta in deltas:
        c11_t, c12_t, c22_t = assemble_true(theta, delta)
        cv11 = c11_t.evalv(X); cv12 = c12_t.evalv(X); cv22 = c22_t.evalv(X)
        gtrue = lam_max_2x2(cv11, cv12, cv22)
        worst_meas = float(np.max(gtrue))
        d0_upper = mul_up(info["c_d0"], float(delta))
        d1_upper = mul_up(info["c_d1"], float(delta))
        bound = add_up(-info["eta"], add_up(mul_up(info["mu1"], d0_upper),
                       mul_up(mul_up(2.0, info["b_upper"]), d1_upper)))
        rows.append({"delta": float(delta),
                     "measured": worst_meas, "bound": bound})
    return rows


def plot_residual(info, rows, out_path):
    deltas = [r["delta"] for r in rows]
    meas = [r["measured"] for r in rows]
    bnd = [r["bound"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    ax.semilogx(deltas, bnd, color="tab:orange", lw=1.8,
                label=r"theoretical bound $-\eta+\mu_1\delta_0+2b\delta_1$")
    ax.semilogx(deltas, meas, color="tab:blue", lw=1.8,
                label=r"measured $\max_x\lambda_{\max}(C^{\rm true}_\theta(x))$")
    ax.axhline(0.0, color="black", lw=0.7, ls=":")
    ax.axvline(info["delta_star"], color="gray", lw=0.8, ls="--")
    ax.axvspan(min(deltas), info["delta_star"], color="tab:green", alpha=0.10,
               label=r"$\delta<\delta^\star$ (certified budget)")
    ax.set_xlabel(r"residual scale $\delta$")
    ax.set_ylabel("worst value of contraction condition")
    ax.set_title(r"Residual robustness (Theorem 2)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    plt.savefig(out_path)
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=160)
    plt.close()


def find_delta_emp(rows):
    """Largest tested delta with nonpositive grid values before any violation.
    This is an observed sweep endpoint, not a certified robustness threshold.
    """
    last = None
    for r in rows:
        if r["measured"] <= 0.0:
            last = r["delta"]
        else:
            break
    return last


def main():
    print("=== Task 3: residual robustness experiment ===")
    print("selecting a certified candidate ...")
    theta, V, ok = select_certified_theta(seed=0)
    if not ok:
        print("WARNING: chosen seed did not certify under the in-loop oracle; "
              "the experiment still proceeds (residual analysis works for any "
              "candidate with positive eta).")
    info = residual_certified_constants(theta, depth=3, n_grid_eta=4001)
    if info["eta"] <= 0.02:
        print(f"  eta = {info['eta']:.4f} <= 0.02; refining n_grid_eta")
        info = residual_certified_constants(theta, depth=3, n_grid_eta=8001)
    if info["eta"] <= 0:
        raise SystemExit("residual experiment has no positive outward-certified raw margin")
    print(f"  eta      = {info['eta']:.5f}")
    print(f"  b_upper  = {info['b_upper']:.4f}")
    print(f"  mu1      = {info['mu1']:.4f}")
    print(f"  c_d0     = {info['c_d0']:.4f}")
    print(f"  c_d1     = {info['c_d1']:.4f}")
    print(f"  delta*   = {info['delta_star']:.5f}")

    # log-spaced sweep bracketing delta_star
    ds = info["delta_star"]
    deltas = list(np.geomspace(0.05 * ds, 4.0 * ds, 25))
    rows = sweep(theta, info, deltas)
    for r in rows:
        print(f"    delta={r['delta']:.4f}  meas={r['measured']:+.5f}  "
              f"bound={r['bound']:+.5f}")

    # soundness check
    violations = [r for r in rows if r["measured"] > r["bound"] + 1e-9]
    if violations:
        print(f"SOUNDNESS VIOLATION: measured exceeds bound in {len(violations)} rows")
        for r in violations[:5]:
            print(f"  delta={r['delta']:.4f}: measured={r['measured']} > bound={r['bound']}")
        raise SystemExit("residual experiment failed soundness check")
    print("  all measured values below the bound: sampled consistency check PASS")

    delta_emp = find_delta_emp(rows)
    print(f"  largest tested delta with measured grid maximum <= 0: {delta_emp}")

    # plot
    plot_residual(info, rows, os.path.join(PAPER, "fig_residual.pdf"))
    print(f"  wrote {os.path.join(PAPER, 'fig_residual.pdf')}")

    # write json
    out = {
        "verification_backend": backend_info(),
        "outward_rounded": bool(info["outward_rounded"]), "grid_enclosed": bool(info["grid_enclosed"]),
        "budget_outward_rounded": bool(info["budget_outward_rounded"]), "rounding_mode": "outward",
        "eta": info["eta"],
        "b_upper": info["b_upper"],
        "mu1": info["mu1"],
        "c_d0": info["c_d0"],
        "c_d1": info["c_d1"],
        "delta_star": info["delta_star"],
        "delta_emp": delta_emp,
        "delta_emp_ratio": (delta_emp / info["delta_star"]) if delta_emp else None,
        "sweep_endpoint": float(deltas[-1]),
        "failure_threshold_identified": any(r["measured"] > 0 for r in rows),
        "interpretation": "delta_emp is the largest tested scale with nonpositive measured grid values, not a certified robustness threshold.",
        "Lg": info["Lg"],
        "n_grid_eta": info["n_grid_eta"],
        "n_grid_meas": 2001,
        "rows": rows,
    }
    with open(os.path.join(RESULTS, "residual.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"  wrote {os.path.join(RESULTS, 'residual.json')}")


if __name__ == "__main__":
    main()
