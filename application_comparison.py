"""Prespecified matched learning study on a published power-system model.

The PDI arm is an equation-based, common-class/common-verifier adaptation of
Fitzsimmons--Liu, not a reproduction of its original LyZNet/dReal workflow.
All three domains, every seed, failures and positive proof covers are retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import autograd.numpy as anp
from autograd import grad
import numpy as np

import application_baseline as baseline
import application_verifier as verifier
import bench_neural
from rounding import backend_info

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "application_protocol.json"
METHOD_LABELS = {"cegis_hinge": "CEGIS hinge", "fitzsimmons_liu_pdi": "F--L PDI adaptation",
                 "hinge_fixed": "Fixed-data hinge control"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config():
    return json.loads(PROTOCOL.read_text())


def hinge_loss(theta, points, rate, margin):
    c11, c12, c22, trace = verifier.contraction_entries(theta, points, rate)
    _, maximum = baseline.eigenvalues_2x2(c11, c12, c22)
    return anp.mean(anp.maximum((maximum + margin) / trace, 0.0))


def screening_grid(B, count):
    axis = np.linspace(-B, B, count)
    return np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)


def run_one(method, seed, B, cfg):
    started = time.perf_counter()
    theta = np.asarray(bench_neural.init_params(seed, cfg["initialization_scale"]), dtype=float)
    initial_theta = theta.copy()
    # A separate deterministic stream keeps initialization independent of data.
    rng_data = np.random.default_rng(seed + 100_000)
    data = rng_data.uniform(-B, B, (cfg["initial_samples"], 2))
    initial_data_hash = hashlib.sha256(data.tobytes()).hexdigest()
    batch_rng = np.random.default_rng(seed + 200_000)
    objective = baseline.baseline_loss if method == "fitzsimmons_liu_pdi" else hinge_loss
    gradient = grad(lambda p, X: objective(p, X, cfg["target_rate"], cfg["training_raw_margin"]))
    screen = screening_grid(B, cfg["screen_axis_points"])
    trace, learner_s, oracle_s, steps = [], 0.0, 0.0, 0
    proof = None
    failure = None
    for round_index in range(cfg["max_rounds"] + 1):
        before = time.perf_counter()
        result = verifier.adaptive_oracle(
            theta, [-B, -B], [B, B], rate=cfg["target_rate"], alpha=cfg["metric_floor"],
            target_margin=cfg["verification_raw_margin"],
            max_boxes=cfg["verification_max_boxes"], max_depth=cfg["verification_max_depth"],
            check_points=screen)
        elapsed = time.perf_counter() - before
        oracle_s += elapsed
        current_loss = float(objective(theta, data, cfg["target_rate"], cfg["training_raw_margin"]))
        record = {"round": round_index, "steps_completed": steps, "dataset_size": len(data),
                  "loss": current_loss if math.isfinite(current_loss) else None,
                  "certified": bool(result["certified"]), "reason": result["reason"],
                  "counters": result["counters"], "oracle_s": elapsed,
                  "parameter_l2": float(np.linalg.norm(theta))}
        trace.append(record)
        if result["certified"]:
            proof = result
            break
        if round_index == cfg["max_rounds"]:
            failure = result["reason"]
            break
        if method == "cegis_hinge":
            points = np.asarray(result.get("training_points", []), dtype=float).reshape(-1, 2)
            # At most the declared number is added; exact duplicates add no data.
            if len(points):
                data = np.unique(np.concatenate([data, points[:cfg["returned_points"]]]), axis=0)
        begin_training = time.perf_counter()
        first, second = np.zeros_like(theta), np.zeros_like(theta)
        for step in range(1, cfg["training_steps_per_round"] + 1):
            if len(data) > cfg["batch_size"]:
                X = data[batch_rng.choice(len(data), cfg["batch_size"], replace=False)]
            else:
                X = data
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                g = np.asarray(gradient(theta, X))
                first = cfg["adam_beta1"] * first + (1-cfg["adam_beta1"]) * g
                second = cfg["adam_beta2"] * second + (1-cfg["adam_beta2"]) * g*g
                update = (cfg["learning_rate"] * (first/(1-cfg["adam_beta1"]**step)) /
                          (np.sqrt(second/(1-cfg["adam_beta2"]**step)) + cfg["adam_epsilon"]))
                candidate = theta - update
            steps += 1
            if not np.all(np.isfinite(candidate)):
                failure = "nonfinite_optimizer_update"
                break
            theta = candidate
        learner_s += time.perf_counter() - begin_training
        if failure:
            break
    out = {"method": method, "seed": seed, "halfwidth": B, "initial_theta": initial_theta.tolist(),
           "initial_data_sha256": initial_data_hash, "theta": theta.tolist(),
           "certified": proof is not None, "failure_reason": failure,
           "steps_completed": steps, "learner_s": learner_s, "oracle_s": oracle_s,
           "total_s": time.perf_counter()-started, "trace": trace, "certificate": proof}
    if proof is not None:
        out["guarantees"] = verifier.metric_guarantees(theta, [-B, -B], [B, B],
            cfg["target_rate"], cfg["verification_raw_margin"], cfg["metric_floor"])
    return out


def refine_frozen_candidate(run, cfg):
    """Exploratory verifier-only recheck, retaining the primary outcome.

    This larger budget was chosen after preliminary primary outcomes. It is
    not another training arm, a replacement primary endpoint, or a rerun
    selected only for favorable candidates. Every final candidate is rechecked.
    """
    specification = cfg["secondary_refinement"]
    theta = np.asarray(run["theta"], float).copy()
    B = run["halfwidth"]
    before_hash = verifier.candidate_hash(theta, cfg["target_rate"], cfg["metric_floor"])
    before = time.perf_counter()
    result = verifier.adaptive_oracle(theta, [-B,-B], [B,B],
        rate=cfg["target_rate"], alpha=cfg["metric_floor"],
        target_margin=cfg["verification_raw_margin"], max_boxes=specification["max_boxes"],
        max_depth=cfg["verification_max_depth"],
        check_points=screening_grid(B,cfg["screen_axis_points"]))
    elapsed = time.perf_counter()-before
    if not np.array_equal(theta,run["theta"]):
        raise ValueError("frozen candidate changed during verification refinement")
    report = {"exploratory":True,"selected_after_preliminary_outcomes":True,
              "candidate_sha256":before_hash,"certified":bool(result["certified"]),
              "oracle_s":elapsed,"total_primary_plus_refinement_s":run["total_s"]+elapsed,
              "certificate":result}
    if result["certified"]:
        report["guarantees"] = verifier.metric_guarantees(theta,[-B,-B],[B,B],
            cfg["target_rate"],cfg["verification_raw_margin"],cfg["metric_floor"])
    return report


def summarize(runs, cfg):
    rows = []
    for B in cfg["domain_halfwidths"]:
        for method in cfg["methods"]:
            selected = [r for r in runs if r["halfwidth"] == B and r["method"] == method]
            successes = [r for r in selected if r["certified"]]
            failures = {}
            for r in selected:
                if not r["certified"]:
                    reason = r["failure_reason"]
                    failures[reason] = failures.get(reason, 0) + 1
            rows.append({"halfwidth": B, "method": method, "runs": len(selected),
                         "certified": len(successes), "failures": failures,
                         "median_updates_all": float(np.median([r["steps_completed"] for r in selected])),
                         "median_learner_s_all": float(np.median([r["learner_s"] for r in selected])),
                         "median_oracle_s_all": float(np.median([r["oracle_s"] for r in selected])),
                         "median_total_s_all": float(np.median([r["total_s"] for r in selected])),
                         "median_updates_certified": (float(np.median([r["steps_completed"] for r in successes]))
                                                       if successes else None)})
    paired = []
    for B in cfg["domain_halfwidths"]:
        lookup = {(r["method"], r["seed"]): r for r in runs if r["halfwidth"] == B}
        for comparator in [m for m in cfg["methods"] if m != "cegis_hinge"]:
            counts = {"both": 0, "cegis_only": 0, "comparator_only": 0, "neither": 0}
            for seed in cfg["seeds"]:
                a, b = lookup["cegis_hinge", seed]["certified"], lookup[comparator, seed]["certified"]
                counts["both" if a and b else "cegis_only" if a else "comparator_only" if b else "neither"] += 1
            paired.append({"halfwidth": B, "comparator": comparator, **counts})
    refined_rows, refined_paired = [], []
    for B in cfg["domain_halfwidths"]:
        for method in cfg["methods"]:
            selected = [r for r in runs if r["halfwidth"] == B and r["method"] == method]
            if not all("refinement" in r for r in selected):
                raise ValueError("frozen-candidate diagnostic missing")
            failures = {}
            for r in selected:
                refinement = r["refinement"]
                if not refinement["certified"]:
                    reason = refinement["certificate"]["reason"]
                    failures[reason] = failures.get(reason,0)+1
            refined_rows.append({"halfwidth":B,"method":method,"runs":len(selected),
                "certified":sum(r["refinement"]["certified"] for r in selected),
                "additional_certified":sum(r["refinement"]["certified"] and not r["certified"] for r in selected),
                "failures":failures,
                "median_additional_oracle_s_all":float(np.median([r["refinement"]["oracle_s"] for r in selected])),
                "median_primary_plus_refinement_s_all":float(np.median([r["refinement"]["total_primary_plus_refinement_s"] for r in selected]))})
        lookup = {(r["method"],r["seed"]):r for r in runs if r["halfwidth"] == B}
        for comparator in [m for m in cfg["methods"] if m != "cegis_hinge"]:
            counts = {"both":0,"cegis_only":0,"comparator_only":0,"neither":0}
            for seed in cfg["seeds"]:
                a,b = lookup["cegis_hinge",seed]["refinement"]["certified"], lookup[comparator,seed]["refinement"]["certified"]
                counts["both" if a and b else "cegis_only" if a else "comparator_only" if b else "neither"] += 1
            refined_paired.append({"halfwidth":B,"comparator":comparator,**counts})
    return {"rows": rows, "paired": paired,
            "refinement_rows":refined_rows,"refinement_paired":refined_paired}


def _validate_refinement(run, cfg, replay):
    refinement = run.get("refinement")
    if (not isinstance(refinement,dict) or refinement.get("exploratory") is not True
            or refinement.get("selected_after_preliminary_outcomes") is not True
            or type(refinement.get("certified")) is not bool):
        raise ValueError("missing or mislabeled exploratory verification refinement")
    theta, B = np.asarray(run["theta"]),run["halfwidth"]
    expected_hash = verifier.candidate_hash(theta,cfg["target_rate"],cfg["metric_floor"])
    if refinement.get("candidate_sha256") != expected_hash:
        raise ValueError("verification refinement changed the frozen candidate")
    cert = refinement["certificate"]
    if (cert.get("proof",{}).get("candidate_sha256") != expected_hash
            or cert.get("certified") != refinement["certified"]
            or cert.get("status") != ("certified" if refinement["certified"] else "inconclusive")
            or cert.get("domain_lower") != [-B,-B] or cert.get("domain_upper") != [B,B]
            or cert.get("quadratic_region") is not None or cert.get("rate") != cfg["target_rate"]
            or cert.get("alpha") != cfg["metric_floor"]
            or cert.get("target_margin") != cfg["verification_raw_margin"]
            or cert.get("budget") != {"max_boxes":cfg["secondary_refinement"]["max_boxes"],"max_depth":cfg["verification_max_depth"]}
            or cert.get("counters",{}).get("nominal_point_evaluations") != cfg["screen_axis_points"]**2):
        raise ValueError("refinement changed candidate, domain, margin, model, screen or declared budget")
    if (not math.isfinite(refinement["oracle_s"]) or refinement["oracle_s"] < 0
            or refinement["total_primary_plus_refinement_s"] != run["total_s"]+refinement["oracle_s"]):
        raise ValueError("refinement timing omits primary or additional work")
    if run["certified"] and not refinement["certified"]:
        raise ValueError("larger fixed-candidate verification budget lost a primary certificate")
    if replay:
        verifier.replay_oracle_certificate(cert,theta)
    if refinement["certified"]:
        expected = verifier.metric_guarantees(theta,[-B,-B],[B,B],
            cfg["target_rate"],cfg["verification_raw_margin"],cfg["metric_floor"])
        if refinement.get("guarantees") != expected:
            raise ValueError("refined quantitative guarantees changed")
    elif "guarantees" in refinement:
        raise ValueError("inconclusive refinement claims quantitative guarantees")


def validate_results(data, replay=True):
    cfg = config()
    if data.get("production_run") is not True:
        raise ValueError("pilot or unmarked results cannot enter the production comparison")
    if data.get("protocol_sha256") != digest(PROTOCOL) or data.get("config") != cfg:
        raise ValueError("application protocol changed")
    if data.get("baseline") != baseline.protocol_description():
        raise ValueError("application baseline provenance changed")
    refinement_specification = cfg.get("secondary_refinement",{})
    if (refinement_specification.get("exploratory") is not True
            or refinement_specification.get("selected_after_preliminary_outcomes") is not True
            or type(refinement_specification.get("max_boxes")) is not int
            or refinement_specification["max_boxes"] <= cfg["verification_max_boxes"]):
        raise ValueError("missing transparent exploratory refinement protocol")
    if data.get("verification_backend", {}).get("outward_rounded") is not True:
        raise ValueError("application verification backend missing")
    expected = {(m, s, B) for m in cfg["methods"] for s in cfg["seeds"] for B in cfg["domain_halfwidths"]}
    runs = data["runs"]
    keys = [(r["method"], r["seed"], r["halfwidth"]) for r in runs]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("missing, repeated or unplanned application runs")
    for r in runs:
        theta = np.asarray(r["theta"])
        if theta.shape != (cfg["parameters"],) or not np.all(np.isfinite(theta)):
            raise ValueError("invalid saved application parameters")
        if (type(r["steps_completed"]) is not int
                or not 0 <= r["steps_completed"] <= cfg["max_rounds"]*cfg["training_steps_per_round"]):
            raise ValueError("optimizer budget exceeded")
        if not 1 <= len(r["trace"]) <= cfg["max_rounds"]+1:
            raise ValueError("oracle call budget exceeded")
        if not all(math.isfinite(r[k]) and r[k] >= 0 for k in ("learner_s", "oracle_s", "total_s")):
            raise ValueError("invalid application timings")
        if type(r["certified"]) is not bool:
            raise ValueError("invalid application success flag")
        previous_size = cfg["initial_samples"]
        counter_keys = ("boxes_evaluated", "accepted_leaves", "excluded_leaves", "splits",
                        "maximum_depth_reached", "nominal_point_evaluations")
        for index, record in enumerate(r["trace"]):
            if (record["round"] != index or record["steps_completed"] != index*cfg["training_steps_per_round"]
                    or type(record["dataset_size"]) is not int
                    or not previous_size <= record["dataset_size"] <= previous_size+(cfg["returned_points"] if index else 0)):
                raise ValueError("invalid matched training schedule")
            if r["method"] != "cegis_hinge" and record["dataset_size"] != cfg["initial_samples"]:
                raise ValueError("fixed-data comparator received extra collocations")
            previous_size = record["dataset_size"]
            if (type(record["certified"]) is not bool or (record["certified"] and index != len(r["trace"])-1)
                    or not math.isfinite(record["oracle_s"]) or record["oracle_s"] < 0
                    or not math.isfinite(record["parameter_l2"]) or record["parameter_l2"] < 0):
                raise ValueError("invalid per-round application record")
            counts = record["counters"]
            if (any(type(counts.get(k)) is not int or counts[k] < 0 for k in counter_keys)
                    or counts["boxes_evaluated"] > cfg["verification_max_boxes"]
                    or counts["maximum_depth_reached"] > cfg["verification_max_depth"]
                    or counts["nominal_point_evaluations"] != cfg["screen_axis_points"]**2):
                raise ValueError("application oracle budget or common screen changed")
        if abs(sum(t["oracle_s"] for t in r["trace"])-r["oracle_s"]) > 1e-9*max(1.,r["oracle_s"]):
            raise ValueError("application oracle timing aggregate changed")
        if r["total_s"]+1e-9 < r["oracle_s"]+r["learner_s"]:
            raise ValueError("application total timing omits measured work")
        if r["failure_reason"] == "nonfinite_optimizer_update":
            if not r["trace"][-1]["steps_completed"] < r["steps_completed"] <= r["trace"][-1]["steps_completed"]+cfg["training_steps_per_round"]:
                raise ValueError("inconsistent numerical-failure update count")
        elif r["steps_completed"] != r["trace"][-1]["steps_completed"]:
            raise ValueError("final optimizer count differs from final verification")
        B, seed = r["halfwidth"], r["seed"]
        initial = bench_neural.init_params(seed, cfg["initialization_scale"])
        points = np.random.default_rng(seed+100_000).uniform(-B, B, (cfg["initial_samples"],2))
        if not np.array_equal(initial, r["initial_theta"]) or hashlib.sha256(points.tobytes()).hexdigest() != r["initial_data_sha256"]:
            raise ValueError("unmatched initialization or collocations")
        if r["certified"]:
            cert = r["certificate"]
            if cert is None or r["failure_reason"] is not None or not r["trace"][-1]["certified"]:
                raise ValueError("inconsistent application success")
            if cert["domain_lower"] != [-B,-B] or cert["domain_upper"] != [B,B]:
                raise ValueError("certificate domain mismatch")
            if cert["target_margin"] != cfg["verification_raw_margin"] or cert["rate"] != cfg["target_rate"]:
                raise ValueError("certificate target mismatch")
            if (cert["alpha"] != cfg["metric_floor"] or cert.get("quadratic_region") is not None
                    or cert["budget"] != {"max_boxes":cfg["verification_max_boxes"],"max_depth":cfg["verification_max_depth"]}
                    or cert["counters"] != r["trace"][-1]["counters"]
                    or cert["proof"].get("candidate_sha256") != verifier.candidate_hash(theta,cfg["target_rate"],cfg["metric_floor"])):
                raise ValueError("certificate differs from the common model class, domain or verifier budget")
            if replay:
                verifier.replay_oracle_certificate(cert, theta)
                expected_guarantees = verifier.metric_guarantees(theta, [-B,-B], [B,B],
                    cfg["target_rate"], cfg["verification_raw_margin"], cfg["metric_floor"])
                if expected_guarantees != r["guarantees"]:
                    raise ValueError("application quantitative guarantees changed")
        else:
            if r["certificate"] is not None or not r["failure_reason"] or any(t["certified"] for t in r["trace"]):
                raise ValueError("inconsistent application failure")
            if r["failure_reason"] != "nonfinite_optimizer_update":
                if (len(r["trace"]) != cfg["max_rounds"]+1
                        or r["failure_reason"] != r["trace"][-1]["reason"]):
                    raise ValueError("unsuccessful run stopped before its prespecified budget")
        _validate_refinement(r,cfg,replay)
    if summarize(runs, cfg) != data["summary"]:
        raise ValueError("application aggregate mismatch")
    expected_energy = {str(B): verifier.certified_energy_level([-B,-B],[B,B]) for B in cfg["domain_halfwidths"]}
    if data["energy_subsets"] != expected_energy:
        raise ValueError("application invariant energy guarantee mismatch")
    return True


def make_figure(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    cfg = data["config"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    colors = ["#3465a4", "#c77b27", "#548b54"]
    Bmax = max(cfg["domain_halfwidths"])
    axis = np.linspace(-Bmax, Bmax, 81)
    delta, omega = np.meshgrid(axis, axis)
    potential = .5*(1-np.cos(delta)) + np.sqrt(3)/2*(np.sin(delta)-delta)
    energy = .5*omega**2+potential
    levels = [data["energy_subsets"][str(B)]["rho_lower"] for B in cfg["domain_halfwidths"]]
    axes[0].contour(delta, omega, energy, levels=levels,
                    colors=["#3465a4", "#555555", "#888888"],
                    linestyles=["solid", "dashed", "dotted"], linewidths=1.5)
    field = verifier.dynamics_nominal(np.stack([delta.ravel(), omega.ravel()],axis=1))
    axes[0].streamplot(axis, axis, field[:,0].reshape(delta.shape), field[:,1].reshape(delta.shape),
                       density=.8, color="#cccccc", linewidth=.6, arrowsize=.7)
    axes[0].plot([0],[0],"ko",markersize=4)
    axes[0].set(xlabel=r"Rotor-angle deviation $\delta$", ylabel=r"Speed deviation $\omega$",
                title="Invariant energy regions", xlim=(-Bmax,Bmax), ylim=(-Bmax,Bmax))
    axes[0].set_aspect("equal")
    handles = [plt.Line2D([0],[0], color=c, linestyle=style, label=f"B = {B:g}")
               for B,c,style in zip(cfg["domain_halfwidths"], ["#3465a4","#555555","#888888"],
                                    ["solid","dashed","dotted"])]
    axes[0].legend(handles=handles,fontsize=8,loc="upper left")
    positions = np.arange(len(cfg["domain_halfwidths"]))
    width = .24
    for k, (method,color) in enumerate(zip(cfg["methods"],colors)):
        rows = [r for r in data["summary"]["rows"] if r["method"]==method]
        refined = [r for r in data["summary"]["refinement_rows"] if r["method"]==method]
        initial = np.asarray([r["certified"] for r in rows])
        added = np.asarray([r["certified"] for r in refined])-initial
        x = positions+(k-1)*width
        axes[1].bar(x,initial,width,color=color,label=METHOD_LABELS[method])
        axes[1].bar(x,added,width,bottom=initial,facecolor="white",edgecolor=color,hatch="///")
    handles, labels = axes[1].get_legend_handles_labels()
    handles.append(Patch(facecolor="white",edgecolor="#555555",hatch="///"))
    labels.append("Added by exploratory refinement")
    axes[1].legend(handles,labels,fontsize=7.5,loc="upper right")
    axes[1].set(xlabel="Box halfwidth B", ylabel="Certified runs", ylim=(0,len(cfg["seeds"])*1.42),
                title="Matched primary and frozen checks")
    axes[1].set_xticks(positions,cfg["domain_halfwidths"])
    axes[1].set_yticks(np.arange(0,len(cfg["seeds"])+1,5))
    axes[1].grid(axis="y",alpha=.2)
    axes[1].set_axisbelow(True)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT/"paper"/("fig_application."+suffix),dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    cfg = config()
    if args.validate:
        data = json.loads((ROOT/"results/application_comparison.json").read_text())
        validate_results(data)
        print("All application records and positive proof covers replayed")
        return
    production = not args.pilot
    active = dict(cfg)
    if args.pilot:
        active.update(seeds=cfg["pilot_seeds_excluded_from_analysis"], max_rounds=1,
                      training_steps_per_round=30)
    runs = []
    for B in active["domain_halfwidths"]:
        for seed in active["seeds"]:
            methods = active["methods"] if seed % 2 == 0 else list(reversed(active["methods"]))
            for method in methods:
                result = run_one(method, seed, B, active)
                runs.append(result)
                print(f"B={B:g} seed={seed:02d} {method}: certified={result['certified']} "
                      f"steps={result['steps_completed']} reason={result['failure_reason']} "
                      f"seconds={result['total_s']:.2f}", flush=True)
    # Keep the original matched comparison intact and freeze every final
    # candidate before this separately labeled, post hoc budget diagnostic.
    for run in runs:
        run["refinement"] = refine_frozen_candidate(run,active)
        refinement = run["refinement"]
        print(f"Exploratory frozen refinement B={run['halfwidth']:g} seed={run['seed']:02d} "
              f"{run['method']}: certified={refinement['certified']} "
              f"reason={refinement['certificate']['reason']} seconds={refinement['oracle_s']:.2f}",flush=True)
    data = {"schema_version":1, "config":active, "protocol_sha256":digest(PROTOCOL),
            "verification_backend":backend_info(), "baseline":baseline.protocol_description(),
            "runs":runs, "summary":summarize(runs, active),
            "energy_subsets": {str(B):verifier.certified_energy_level([-B,-B],[B,B])
                               for B in active["domain_halfwidths"]},
            "production_run":production}
    if production:
        validate_results(data)
        destination = ROOT/"results/application_comparison.json"
    else:
        destination = ROOT/"tmp/application_pilot.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    if production:
        make_figure(data)
    print("Wrote", destination)


if __name__ == "__main__":
    main()
