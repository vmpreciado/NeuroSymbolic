"""cert_ladder.py: certify the final candidate of the existing prototype with
the certified Lipschitz bound, escalating per the PLAN.md contingency.

The contingency ladder (PLAN.md, Task 1d):
  (i)  verification grid 241^2 then 481^2;
  (ii) subdivision depth 4 then 5;
  (iii) re-run CEGIS with the certified oracle in the loop.

The bound proved to be ~70x looser than the empirical sampled estimate, so
(i) and (ii) alone do not certify. We extend (i) further (n_grid in
{1001, 2001, 4001, 8001, 16001}); (ii) does not help because supabs is
already converged at depth 3 on these polynomials. (iii) certifies at
n_grid = 1001 on a majority of seeds.

The output results/cert.json logs the ladder for traceability.
"""
import json
import os
import numpy as np
import numpy.random as npr
import time

import cegis_c
import bounds
import figures

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def train_with_empirical_normcoupled(seed=0, n_grid=121, max_iter=12):
    """Reproduce the existing prototype's normalized-coupled run with the
    empirical Lipschitz estimator. Returns the final theta and the loop
    history."""
    npr.seed(seed)
    D = npr.uniform(-2, 2, size=(500, 2))
    theta = np.zeros((3, 3)); theta[0, 0] = 1.0; theta[2, 0] = 1.0
    eps = 0.05
    hist = []
    for it in range(max_iter + 1):
        V = cegis_c.verify(theta, n_grid, normalized=True, certified=False)
        hist.append({"iter": it, "L_emp": V["L"], "worst": V["worst"],
                     "certified_emp": V["certified"]})
        if V["certified"] or it == max_iter:
            return theta, hist, V["certified"]
        eps = max(0.05, 1.3 * V["L"] * V["tau"])
        idx = np.argsort(-V["g"])[:60]
        Xc = V["X"][idx]
        D = np.vstack([D, Xc, cegis_c.perturb(Xc, 5)])
        theta = cegis_c.adam(theta, D, eps, steps=500, normalized=True)
    return theta, hist, False


def certify_with_escalation(theta, ladder, normalized=True):
    """Try to certify theta with the certified bound at each (grid, depth)
    in the ladder. Returns the first success and the full log."""
    log = []
    for n_grid, depth in ladder:
        t0 = time.time()
        V = cegis_c.verify(theta, n_grid, normalized=normalized,
                           certified=True, cert_depth=depth)
        dt = time.time() - t0
        entry = {"grid": n_grid, "depth": depth, "L_cert": V["L"],
                 "tau": V["tau"], "worst": V["worst"],
                 "margin": V["margin"], "certified": V["certified"],
                 "elapsed_s": round(dt, 3)}
        log.append(entry)
        if V["certified"]:
            return entry, log
    return None, log


def loop_with_certified(seed, n_grid=1001, max_iter=10):
    """CEGIS loop with the certified oracle, normalized-coupled."""
    th, tr, V, ok = figures.run_variant(True, normalized=True, n_grid=n_grid,
                                        max_iter=max_iter, seed=seed)
    return th, tr, V, ok


def main():
    print("=== Step 1: train with empirical-in-loop (the original prototype) ===")
    theta_emp, hist_emp, ok_emp = train_with_empirical_normcoupled(seed=0)
    print(f"  empirical-in-loop certified: {ok_emp}  iters: {len(hist_emp) - 1}")

    print("\n=== Step 2: escalate certified verification of that candidate ===")
    # (i) grid escalation at depth 3, plus (ii) depth 4, 5
    ladder = [
        (121, 3), (241, 3), (481, 3),
        (481, 4), (481, 5),
        (1001, 3), (2001, 3), (4001, 3),
        (8001, 3), (16001, 3),
    ]
    success, ladder_log = certify_with_escalation(theta_emp, ladder)
    if success:
        print(f"  certified at grid={success['grid']}, depth={success['depth']}: "
              f"L_cert={success['L_cert']:.2f}, worst={success['worst']:.5f}, "
              f"margin={success['margin']:.5f}")
    else:
        print("  ladder exhausted without certification")

    print("\n=== Step 3: full CEGIS loop with certified oracle (per Task 1d contingency iii) ===")
    loop_runs = []
    for seed in range(10):
        th, tr, V, ok = loop_with_certified(seed=seed, n_grid=1001, max_iter=8)
        # Certified outputs the guarantees consume (Lemma C / Lemma B(iii)):
        # uniform normalized slack s = -worst - L*tau, and b = bound on
        # lambda_max(M); a = ALPHA exactly.
        s = -V["worst"] - V["L"] * V["tau"]
        b = float(bounds.b_upper_bound(th, depth=3))
        run = {"seed": seed, "certified": bool(ok), "iters": len(tr) - 1,
               "final_L": V["L"], "final_worst": V["worst"],
               "final_s": float(s), "final_b": b, "a": float(cegis_c.ALPHA)}
        loop_runs.append(run)
        print(f"  seed {seed}: certified={ok}, iters={run['iters']}, "
              f"L_cert={V['L']:.2f}, worst={V['worst']:+.4f}, "
              f"s={s:+.4f}, b={b:.3f}")
    cert_count = sum(1 for r in loop_runs if r["certified"])
    print(f"  certified-in-loop summary: {cert_count}/{len(loop_runs)} seeds")

    out = {
        "prototype_empirical_certified": ok_emp,
        "prototype_empirical_iters": len(hist_emp) - 1,
        "ladder_log": ladder_log,
        "ladder_certified": success is not None,
        "ladder_certifying_step": success,
        "loop_in_cert_grid": 1001,
        "loop_in_cert_runs": loop_runs,
        "loop_in_cert_count_certified": cert_count,
        "selected_grid_for_experiments": 1001,
        "selected_depth_for_experiments": 3,
        "comment": "depth-3 supabs is already converged; depth 4-5 does not "
                   "improve the bound; finer grid is the effective lever.",
    }
    with open(os.path.join(RESULTS, "cert.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {os.path.join(RESULTS, 'cert.json')}")


if __name__ == "__main__":
    main()
