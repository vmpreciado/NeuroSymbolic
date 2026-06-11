"""Figures for the CEGIS-C paper prototype."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import autograd.numpy as np
import numpy.random as npr
from cegis_c import (lam_max_C, adam, verify, perturb, grid,
                     EPS, BOX)

def identity_init():
    th = np.zeros((3, 3))
    th[0, 0] = 1.0
    th[2, 0] = 1.0
    return th

def run_variant(couple_margin, normalized=False, n_grid=121, max_iter=12,
                k_worst=60, m_pert=5, seed=0):
    """CEGIS loop, verify-first, per-iteration trace of the certifiability
    gap = worst test value minus required margin (gap <= 0 certifies)."""
    npr.seed(seed)
    D = npr.uniform(-BOX, BOX, size=(500, 2))
    theta = identity_init()
    eps = EPS
    trace = []
    for it in range(max_iter + 1):
        V = verify(theta, n_grid, normalized=normalized)
        trace.append({"iter": it, "worst": V["worst"], "margin": V["margin"],
                      "gap": V["worst"] - V["margin"],
                      "viol": int(V["viol"].sum()), "eps": eps, "L": V["L"]})
        if V["certified"] or it == max_iter:
            return theta, trace, V, V["certified"]
        if couple_margin:
            eps = max(EPS, 1.3 * V["L"] * V["tau"])
        idx = np.argsort(-V["g"])[:k_worst]
        Xc = V["X"][idx]
        D = np.vstack([D, Xc, perturb(Xc, m_pert)])
        theta = adam(theta, D, eps, steps=500, normalized=normalized)
    return theta, trace, V, V["certified"]

# ---- run both variants ----
# coarse grid (121^2): required margin L*tau exceeds fixed eps, so the
# fixed-margin learner is in its stall regime
th_fix, tr_fix, V_fix, ok_fix = run_variant(False)
th_nai, tr_nai, V_nai, ok_nai = run_variant(True, normalized=False)
th_cpl, tr_cpl, V_cpl, ok_cpl = run_variant(True, normalized=True)
print("fixed:", ok_fix, "| naive coupled:", ok_nai,
      "| normalized coupled:", ok_cpl, f"({len(tr_cpl)-1} iters)")
print("naive-coupled L by iter:", [round(t["L"], 1) for t in tr_nai])

# ---- Figure 1: heatmaps, identity metric vs certified metric ----
n = 301
X, s = grid(n)
g0 = lam_max_C(identity_init(), X).reshape(n, n)
g1 = lam_max_C(th_cpl, X).reshape(n, n)
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
plt.savefig("fig_heatmaps.pdf"); plt.savefig("fig_heatmaps.png", dpi=160)
plt.close()

# ---- Figure 2: stall vs margin-coupled traces ----
fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
for tr, lab, c, mk in [
        (tr_fix, r"fixed $\varepsilon$ (stalls)", "tab:red", "o"),
        (tr_nai, r"coupled $\varepsilon_k$, unnormalized (scale chase)",
         "tab:orange", "^"),
        (tr_cpl, r"coupled $\varepsilon_k$, trace-normalized (certifies)",
         "tab:blue", "s")]:
    it = [t["iter"] for t in tr]
    gap = [max(t["gap"], 1e-6) for t in tr]
    ax.semilogy(it, gap, marker=mk, color=c, label=lab, lw=1.5, ms=4)
ax.set_xlabel("CEGIS iteration")
ax.set_ylabel(r"certifiability gap (certified $\Leftrightarrow$ gap $\leq 0$)")
ax.legend(fontsize=8, loc="lower left")
plt.savefig("fig_traces.pdf"); plt.savefig("fig_traces.png", dpi=160)
plt.close()

# ---- Figure 3: cost of the grid oracle vs dimension ----
# Grid spacing must satisfy L * tau_cov <= slack, tau_cov = h*sqrt(n)/2,
# so h <= 2*slack/(L*sqrt(n)) and N = (4/h + 1)^n points on [-2,2]^n.
# Use measured L and the achieved interior slack of the certified metric.
Lhat = V_cpl["L"]
from cegis_c import lam_max_C_norm
slack = -float(np.max(lam_max_C_norm(th_cpl, grid(301)[0])))  # normalized slack
dims = np.arange(2, 9)
pts = [( (2 * BOX) / (2 * slack / (Lhat * np.sqrt(d))) + 1) ** d for d in dims]
fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
ax.semilogy(dims, pts, "o-", color="tab:purple")
ax.axhline(1e9, color="gray", ls=":", lw=1)
ax.text(2.1, 2e9, "$10^9$ grid points", fontsize=8, color="gray")
ax.set_xlabel(r"state dimension $n$")
ax.set_ylabel("grid points required for certification")
ax.set_title(f"grid oracle cost ($\\hat L={Lhat:.1f}$, slack $={slack:.3f}$)",
             fontsize=10)
plt.savefig("fig_gridcost.pdf"); plt.savefig("fig_gridcost.png", dpi=160)
plt.close()

print("slack:", round(slack, 4), " L:", round(Lhat, 2))
print("grid points by dim:", {int(d): f"{p:.2e}" for d, p in zip(dims, pts)})
print("figures written")
