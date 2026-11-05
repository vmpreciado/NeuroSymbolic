"""smoke_p1.py: P1 wiring validation (not a paper claim).

Regenerates a stored certified normalized-coupled candidate and records, for
that single candidate: the certified Lipschitz constant, the empirical estimate
on the same candidate and their ratio (the conservatism factor), the certified
uniform normalized slack s (Lemma C), the certified class bound b and the
residual constant mu_1 (Lemma B), and wall-clock. Output feeds
results/smoke_p1.json and, through fill_tex.py, paper/numbers.tex.

Candidate selection: results/gridcost.json lists the seeds whose
normalized-coupled run certified on this machine. That JSON does not store
theta, so we regenerate deterministically with experiments.run_variant (the
exact producer of gridcost.json, using its recorded grid and depth) over those
seeds and take the first that certifies here, recording which. Single-seed runs
are platform dependent (PLAN, CLAUDE.md rule 3), so we confirm certification on
this machine rather than trusting a stored per-seed outcome.

These numbers validate that the certified path is wired correctly. They are a
single-seed diagnostic and never appear as a claim about loop behavior in the
manuscript (CLAUDE.md rule 3).
"""
import json
import os
import time

import cegis_c
import bounds
import experiments
from cegis_c import lipschitz_estimate, verify
from bounds import _empirical_L_gt
from rounding import backend_info

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def _source_configuration():
    with open(os.path.join(RESULTS, "gridcost.json")) as f:
        gc = json.load(f)
    with open(os.path.join(RESULTS, "traces.json")) as f:
        tr = json.load(f)
    cfg = tr["config"]
    n_grid, depth = int(cfg["grid_operating"]), int(cfg["depth"])
    if (n_grid != gc["config"]["grid_operating"]
            or depth != gc["config"]["depth"]):
        raise ValueError("traces and gridcost configurations disagree")
    seeds = [int(r["seed"]) for r in gc["ratios"]]
    if not seeds or n_grid < 2 or depth < 0:
        raise ValueError("missing certified seeds or invalid verification configuration")
    return n_grid, depth, int(cfg["max_iter"]), seeds


def main():
    n_grid, depth, max_iter, seeds = _source_configuration()

    # Regenerate the first candidate that certifies on this machine.
    th = tr = V = None
    seed = None
    for cand in seeds:
        th_c, tr_c, V_c, ok_c = experiments.run_variant(
            True, True, seed=cand, n_grid=n_grid, max_iter=max_iter, depth=depth)
        if ok_c:
            th, tr, V, seed = th_c, tr_c, V_c, cand
            break
    if th is None:
        raise SystemExit("no candidate certified on this machine; check "
                         "experiments configuration before the P1 smoke run")

    # Certified verification of that candidate (timed).
    t0 = time.time()
    Vc = verify(th, n_grid, normalized=True, certified=True, cert_depth=depth)
    t_verify = time.time() - t0
    L_cert = float(Vc["L"])
    tau = float(Vc["tau"])
    worst = float(Vc["worst"])
    s = float(Vc["uniform_slack"])                 # outward lower uniform slack
    if not Vc["certified"] or not (s >= 0.0):
        raise SystemExit("smoke re-verification failed at the recorded operating grid")

    # Empirical estimator on the same candidate: raw (no safety factor) and
    # the coded 1.5x diagnostic. Conservatism = certified / raw empirical.
    raw_emp = float(_empirical_L_gt(th, n_grid=401, normalized=True))
    emp_1p5 = float(lipschitz_estimate(th, normalized=True))
    conservatism = L_cert / raw_emp if raw_emp > 0 else None

    # Class and residual constants (Lemma B), timed as one block.
    t1 = time.time()
    b = float(bounds.b_upper_bound(th, depth=depth))
    mu1 = float(bounds.mu1_upper_bound(th, depth=depth))
    t_bounds = time.time() - t1

    out = {
        "verification_backend": backend_info(),
        "outward_rounded": bool(Vc["outward_rounded"]), "grid_enclosed": bool(Vc["grid_enclosed"]),
        "rounding_mode": "outward",
        "candidate": {
            "seed": seed, "n_grid": n_grid, "depth": depth,
            "certified": bool(Vc["certified"]), "iters": len(tr) - 1,
            "grid_enclosed": bool(Vc["grid_enclosed"]),
            "outward_rounded": bool(Vc["outward_rounded"]),
            "reproduced_from": ("experiments.run_variant(True, True, "
                                "seed=%d, n_grid=%d, max_iter=%d, depth=%d); "
                                "first certifying seed from results/gridcost.json"
                                % (seed, n_grid, max_iter, depth)),
        },
        "L_cert": L_cert,
        "L_emp_raw": raw_emp,
        "L_emp_1p5": emp_1p5,
        "conservatism_factor": (float(conservatism)
                                if conservatism is not None else None),
        "tau": tau,
        "worst_gt": worst,
        "s": float(s),
        "a": float(cegis_c.ALPHA),
        "b": b,
        "mu1": mu1,
        "wall_verify_s": float(t_verify),
        "wall_bounds_s": float(t_bounds),
    }

    with open(os.path.join(RESULTS, "smoke_p1.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    print(f"candidate: seed={seed} n_grid={n_grid} depth={depth} "
          f"certified={out['candidate']['certified']} iters={out['candidate']['iters']}")
    print(f"L_cert            = {L_cert:.4f}")
    print(f"L_emp_raw         = {raw_emp:.4f}")
    print(f"L_emp_1p5         = {emp_1p5:.4f}")
    print(f"conservatism (cert/raw_emp) = {conservatism:.1f}x")
    print(f"tau               = {tau:.6f}")
    print(f"worst_gt          = {worst:+.6f}")
    print(f"s (certified slack) = {s:+.6f}")
    print(f"a                 = {out['a']:.4f}")
    print(f"b                 = {b:.4f}")
    print(f"mu1               = {mu1:.4f}")
    print(f"wall_verify_s     = {t_verify:.3f}")
    print(f"wall_bounds_s     = {t_bounds:.3f}")
    print(f"wrote {os.path.join(RESULTS, 'smoke_p1.json')}")


if __name__ == "__main__":
    main()
