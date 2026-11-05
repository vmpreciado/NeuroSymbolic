"""Generate complete certificate/reporting evidence, then render it through fill_tex.

The report preserves per-candidate relationships. Componentwise medians are
descriptive statistics, never substituted together into a certificate.
"""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import numpy as np
import bounds
import cegis_c
import bench_pendulum
from rounding import (backend_info, mul_up, mul_down, div_down, div_up,
                      add_down, add_up, sqrt_up, sub_down, float_down, grid_radius)
from fractions import Fraction

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"


def load(name):
    return json.loads((RESULTS / name).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def certificate_row(study, seed, run, config, grid, rate, metric_box):
    t = run["trace"][-1]
    if (not run["certified"] or not t["grid_enclosed"] or not t.get("outward_rounded")
            or not 0 <= t["uniform_slack"] <= sub_down(-t["worst"],mul_up(t["L"],t["tau"]))):
        raise ValueError("certificate table requires a positive outward grid check")
    a = float(config.get("metric_floor", 0.1))
    if study == "Neural":
        b = run["b"]
        trace_lower = mul_down(2.0, a)
        eta = run["raw_margin_lower"]
        rate_lower = run["rate_lower"]
        overshoot = run["overshoot_upper"]
        params = run["parameters"]
    else:
        params = run["theta"]
        theta = np.asarray(params)
        b = bounds.b_upper_bound(theta, depth=3, B=metric_box)
        trace_lower = bounds.v_min_bound(theta, depth=3, B=metric_box)
        eta = max(0.0,mul_down(trace_lower, t["uniform_slack"]))
        rate_lower = max(rate,add_down(rate, div_down(eta, mul_up(2.0, b))))
        overshoot = sqrt_up(div_up(b, a))
    return dict(study=study, seed=int(seed), grid_axis=grid, N=grid ** 2,
                tau=t["tau"], L=t["L"], Ltau=mul_up(t["L"], t["tau"]),
                grid_slack_lower=-t["worst"], uniform_slack=t["uniform_slack"],
                trace_lower=trace_lower, eta=eta, a=a, b=b,
                rate_lower=rate_lower, overshoot_upper=overshoot,
                metric_sha256=digest(params), depth=3,
                margin_method="certified trace lower bound times certified normalized slack",
                outward_rounded=True, grid_enclosed=True)


def stats(values):
    values = [float(v) for v in values]
    if not values or not all(np.isfinite(values)):
        raise ValueError("nonfinite or empty report statistics")
    return dict(median=float(np.median(values)), minimum=min(values), maximum=max(values))


def timing_rows(data):
    found = []
    def visit(value):
        if isinstance(value, dict):
            if "iter" in value and "train_s" in value:
                if value["iter"] > 0:
                    found.append(value)
            else:
                for item in value.values(): visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
    visit(data["runs"])
    return found


def coordinate_margin(theta, axis_points, depth=3):
    """Directional quotient bounds paired with actual axis-grid coverage."""
    model = bounds.assemble_exact(theta)
    trace_lower = bounds.v_min_bound(theta, depth, P=model)
    G = bounds.sup_abs_gt_bound(theta, depth, P=model)
    derivatives=[]
    for k in range(2):
        values=[bounds.supabs(p,depth) for p in model["dc"][k]]
        D=bounds._norm_upper(values,[1,2,1])
        V=bounds.supabs(model["dv"][k],depth)
        derivatives.append(div_up(add_up(D,mul_up(G,V)),trace_lower))
    radius=grid_radius(np.linspace(-cegis_c.BOX,cegis_c.BOX,axis_points),dimension=1)
    margin=0.0
    for E in derivatives: margin=add_up(margin,mul_up(E,radius))
    return dict(directional_lipschitz_upper=derivatives,axis_radius_upper=radius,
                coordinate_margin_upper=margin,outward_rounded=True)


def refinement_study(tr):
    selected=sorted((int(seed),run) for seed,run in tr["regimes"]["certify"].items()
                    if run["certified"])[:3]
    records=[]
    for seed,run in selected:
        theta=np.asarray(run["theta"])
        steps=[]
        for axis in (121,241,481,961,1921,2001):
            v=cegis_c.verify(theta,axis,normalized=True,certified=True,cert_depth=3)
            steps.append(dict(axis_points=axis,nominal_evaluations=axis**2,
                              outward_evaluations=axis**2 if v["grid_enclosed"] else 0,
                              certified=bool(v["certified"]),grid_enclosed=v["grid_enclosed"],
                              L=v["L"],tau=v["tau"],worst=v["worst"],
                              uniform_slack=v["uniform_slack"],outward_rounded=v["outward_rounded"]))
            if v["certified"]: break
        if not steps[-1]["certified"]:
            raise ValueError("fixed-candidate refinement failed at original certified resolution")
        records.append(dict(seed=seed,metric_sha256=digest(run["theta"]),steps=steps,
                            coordinate_bound_at_final_grid=coordinate_margin(theta,steps[-1]["axis_points"])))
    return dict(policy="freeze candidate; refine within a prespecified axis-count budget; no learner updates",
                axis_schedule=[121,241,481,961,1921,2001],records=records,
                scope="diagnostic for known strictly certifiable candidates; not a termination guarantee for a moving learner")


def generate():
    tr, pe, ne = load("traces.json"), load("pendulum.json"), load("neural.json")
    rows = []
    specs = [("Polynomial", tr["regimes"]["certify"], tr["config"],
              tr["config"]["grid_operating"], cegis_c.RATE, cegis_c.BOX),
             ("Pendulum", pe["runs"], pe["config"], pe["n_grid"], pe["rate"], pe["box"]),
             ("Neural", ne["runs"], ne["config"], ne["op_grid"],
              ne["config"]["target_rate"], cegis_c.BOX)]
    for study, runs, config, grid, rate, box in specs:
        rows += [certificate_row(study, seed, run, config, grid, rate, box)
                 for seed, run in runs.items() if run["certified"]]
    summaries = {}
    for study, *_ in specs:
        subset = [r for r in rows if r["study"] == study]
        summaries[study] = {k: stats([r[k] for r in subset]) for k in
                            ("L", "Ltau", "uniform_slack", "eta", "b", "rate_lower", "overshoot_upper")}
    timings = {}
    polynomial_timing=load("timing_polynomial.json")
    timing_specs=[("Polynomial / fixed coarse",{"runs":polynomial_timing["runs"]["stall"]}),
                  ("Polynomial / coupled coarse",{"runs":polynomial_timing["runs"]["diverge"]}),
                  ("Polynomial / operating",{"runs":polynomial_timing["runs"]["certify"]}),
                  ("Pendulum",load("timing_pendulum.json")),("Neural",load("timing_neural.json"))]
    for study, data in timing_specs:
        rr = timing_rows(data)
        timings[study] = {k: stats([r[k] for r in rr]) for k in
                          ("train_s", "verify_including_bounds_s", "oracle_evaluations")}
        timings[study]["rounds"] = len(rr)
    # Rational analytic witness bounds, converted in the safe direction.
    B, alpha, rate = map(Fraction.from_float, (cegis_c.BOX, cegis_c.ALPHA, cegis_c.RATE))
    eig_lower = Fraction(1, 1) / (2 + 4 * B * B)
    positive = 2 * (rate - 1 + B)
    threshold = alpha * max(Fraction(0), positive) / (2 * (1 - rate) * eig_lower)
    from rounding import float_up
    exceptional = {regime: [{"seed": int(s), "certified": r["certified"],
                             "diverged": r["diverged"], "rounds": len(r["trace"])-1}
                            for s, r in tr["regimes"][regime].items()
                            if (r["certified"] if regime == "stall" else not r["certified"])]
                   for regime in ("stall", "certify")}
    cpu_model=platform.processor()
    if platform.system() == "Darwin":
        cpu=subprocess.run(["sysctl","-n","machdep.cpu.brand_string"],capture_output=True,text=True,check=False)
        if cpu.returncode == 0: cpu_model=cpu.stdout.strip()
    out = dict(schema_version=1, verification_backend=backend_info(), certificates=rows,
               summaries=summaries, timings=timings, exceptional_seeds=exceptional,
               representative_rule="smallest certified seed in each study",
               aggregation="componentwise descriptive summaries; use each complete row for guarantees",
               machine=dict(platform=platform.platform(), machine=platform.machine(),
                            cpu_model=cpu_model, python=platform.python_version(),
                            thread_environment=load("run_manifest.json")["environment"]["thread_environment"]),
               pullback=dict(min_eigenvalue_lower=float_down(eig_lower),
                             c_strict_threshold_upper=float_up(threshold)),
               protocol=dict(polynomial_parameters=9, neural_parameters=39, neural_outputs=3,
                             perturbation_std=tr["config"]["perturbation_std"],
                             pendulum_perturbation_std=pe["config"]["perturbation_std"],
                             neural_perturbation_std=ne["config"]["perturbation_std"],
                             optimizer_state=tr["config"]["optimizer_state"]),
               residual_provenance=load("residual.json").get("provenance", {}))
    out["refinement"] = refinement_study(tr)
    out["input_sha256"] = {name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
                            for name in ("traces.json", "pendulum.json", "neural.json", "residual.json",
                                         "timing_polynomial.json", "timing_pendulum.json", "timing_neural.json",
                                         "revision_experiments.json", "revision_extensions.json")}
    (RESULTS / "revision_report.json").write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")
    return out


def render(report=None):
    """Called only by fill_tex.py after all result producers have passed."""
    import fill_tex
    r = load("revision_report.json") if report is None else report
    for name, expected in r["input_sha256"].items():
        if hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"report input changed: {name}; regenerate the report")
    f, fu, fd = fill_tex.fmt, fill_tex.fmt_up, fill_tex.fmt_down
    macros = dict(revPerturbStd=f(r["protocol"]["perturbation_std"]),
                  revPullbackMinEigLower=fd(r["pullback"]["min_eigenvalue_lower"]),
                  revPullbackThreshold=fu(r["pullback"]["c_strict_threshold_upper"]))
    macros["revOtherPerturbStd"] = f(r["protocol"]["pendulum_perturbation_std"])
    provenance=r["residual_provenance"]
    macros.update(revResidualDepth=f(provenance["depth"]),
                  revResidualGridAxis=f(provenance["n_grid_eta"]),
                  revResidualSeed=f(provenance["candidate_seed"]))
    ext=load("revision_extensions.json")
    ec=ext["config"]
    macros["revExtensionSettings"]=("The new benchmark uses $["+f(ec["domain_3d"][0])+","+
        f(ec["domain_3d"][1])+r"]^3$, rate $"+f(ec["rate"])+r"$, floor $"+f(ec["alpha"])+
        "$ and $"+f(ec["n_seeds"])+"$ seeds. Each retraining round has at most $"+
        f(ec["steps_per_round_3d"])+"$ Adam steps, with $"+f(ec["initial_samples_3d"])+
        "$ initial samples and at most $"+f(ec["max_retraining_rounds_3d"])+
        "$ retraining rounds. The oracle uses at most $"+f(ec["adaptive_budget"]["max_boxes"])+
        "$ boxes and depth $"+f(ec["adaptive_budget"]["max_depth"])+
        "$ per call, with target raw margin $"+fd(ec["adaptive_budget"]["target_margin"])+"$.")
    macros["revExtensionSettings"] += (" The learner uses a fixed normalized hinge margin $"+
        f(ec["training_margin_3d"])+"$ and learning rate $"+f(ec["learning_rate"])+
        "$; it does not use the grid loop's coupled margin. Zero-loss retraining that leaves the parameters unchanged stops an inconclusive run.")
    fitting=ext["learned_surrogate"]["fitting"]
    macros["revSurrogateSettings"]=("Quadratic regression uses $"+f(fitting["trajectory_count"])+
        "$ noiseless trajectories with $"+f(fitting["times_per_trajectory"])+
        "$ sampled times each; central differences use time step $"+f(fitting["derivative_dt"])+
        "$. The basis contains every monomial of total degree at most two.")
    sp=ext["learned_surrogate"]["metric_learning"]["protocol"]
    macros["revSurrogateSettings"] += (" Its metric learner uses a fixed normalized hinge margin $"+
        f(sp["margin"])+"$, $"+f(sp["initial_samples"])+"$ initial samples, at most $"+
        f(sp["steps_per_round"])+"$ Adam steps per round, learning rate $"+f(sp["learning_rate"])+
        "$ and at most $"+f(sp["max_rounds"])+"$ retraining rounds; the same unchanged-parameter stopping rule applies.")
    ablation=load("revision_experiments.json")
    macros["revBoundedRadius"]=("The radius is computed before the runs as four times the Frobenius norm of the common initial factor coefficients; its stored value is $R\\approx "+
                                f(ablation["config"]["radius"],sig=6)+"$.")
    sensitivity=ablation["sensitivity"]
    sc=sensitivity["config"]
    macros["revSensitivitySettings"]=("The sensitivity diagnostic uses $"+f(len(sensitivity["seeds"]))+
        "$ seeds, a common $"+f(sc["n_grid"])+r"^2$ grid, and at most $"+f(sc["max_rounds"])+
        "$ retraining rounds with $"+f(sc["training_steps"])+"$ Adam steps each.")
    exceptions=[]
    for regime, label in (("stall", "fixed-margin coarse-grid"), ("certify", "coupled operating-grid")):
        for run in r["exceptional_seeds"][regime]:
            outcome = "certified" if run["certified"] else "met the finite-budget divergence criterion"
            exceptions.append("The "+label+" exception, seed $"+f(run["seed"])+"$, "+outcome+
                              " after $"+f(run["rounds"])+"$ retraining rounds.")
    macros["revLegacySeedsSummary"] = " ".join(exceptions)
    failures=[run for run in ext["three_dimensional"]["runs"] if not run["certified"]]
    failure_sentence=(" The uncertified runs exhaust the box budget and stop after zero-loss retraining leaves the parameters unchanged."
                      if failures and all(run["final_oracle"]["reason"]=="box_budget_exhausted" and
                                           run["stop_reason"]=="zero_loss_unchanged_parameter" for run in failures)
                      else " The saved traces give each uncertified run's stopping reason.")
    macros.update(revAblationSummary="Table~\\ref{tab:ablation} reports every regime, including uncertified runs.",
                  revScaleChaseSummary="Figure~\\ref{fig:scalechase} illustrates the constructed scaling sequence; it does not estimate optimizer failure frequencies.",
                  revExtensionSummary="Table~\\ref{tab:extensions} reports the three-dimensional synthesis and matched-candidate oracle checks."+failure_sentence,
                  revSurrogateSummary="Table~\\ref{tab:extensions} includes global residual bounds for the fitted surrogate and the transferred rate.")
    lines = ["% Generated by fill_tex.py from revision_report.json; do not edit."]
    lines.append("% SHA256 results/revision_report.json: "+hashlib.sha256((RESULTS/"revision_report.json").read_bytes()).hexdigest())
    lines += ["% SHA256 results/"+name+": "+digest for name,digest in sorted(r["input_sha256"].items())]
    lines += [r"\providecommand{\%s}{%s}" % pair for pair in macros.items()]
    (PAPER / "revision_numbers.tex").write_text("\n".join(lines)+"\n")
    tex = [r"% Generated tables. All numerical entries originate in result JSON.",
           "% SHA256 results/revision_report.json: "+hashlib.sha256((RESULTS/"revision_report.json").read_bytes()).hexdigest(),
           r"\begin{table}[t]\centering\caption{Complete representative certificates. The smallest certified seed is used per study. Upper bounds are rounded upward and lower guarantees downward.}\label{tab:certconstants}",
           r"\small\begin{tabular}{lrrr}\hline Quantity & Polynomial & Pendulum & Neural\\\hline"]
    representatives = [min((q for q in r["certificates"] if q["study"] == study), key=lambda q:q["seed"])
                       for study in ("Polynomial", "Pendulum", "Neural")]
    for name, key, formatter in [("Seed", "seed", f), (r"$N_{\rm ax}$", "grid_axis", f),
                                 (r"$\tau$", "tau", fu), (r"$L_h$", "L", fu),
                                 (r"$L_h\tau$", "Ltau", fu), (r"$\underline s$", "uniform_slack", fd),
                                 (r"$\underline v$", "trace_lower", fd), (r"$\eta$", "eta", fd),
                                 (r"$a$", "a", fd), (r"$b$", "b", fu),
                                 (r"$\lambda'$", "rate_lower", fd), (r"$\sqrt{b/a}$", "overshoot_upper", fu)]:
        tex.append(name + " & " + " & ".join("$"+formatter(q[key], sig=8 if key=="rate_lower" else 4)+"$" for q in representatives)+r"\\")
    tex += [r"\hline\end{tabular}\end{table}",
            r"\begin{table}[t]\centering\caption{Certified-seed distributions: median [minimum, maximum]. Each component is summarized separately; these medians do not form a joint certificate. Full per-seed constants are in the reproduction.}\label{tab:certsummaries}",
            r"\scriptsize\begin{tabular}{llrrr}\hline Quantity & & Polynomial & Pendulum & Neural\\\hline"]
    for label, key in [(r"$L_h$", "L"), (r"$L_h\tau$", "Ltau"), (r"$\underline s$", "uniform_slack"),
                       (r"$\eta$", "eta"), (r"$b$", "b"), (r"$\lambda'$", "rate_lower"), (r"$\sqrt{b/a}$", "overshoot_upper")]:
        cells=[]
        for study in ("Polynomial", "Pendulum", "Neural"):
            x=r["summaries"][study][key]
            cells.append("$"+f(x["median"],sig=3)+r"\ ["+f(x["minimum"],sig=3)+", "+f(x["maximum"],sig=3)+"]$")
        tex.append(label+" & & "+" & ".join(cells)+r"\\")
    tex += [r"\hline\end{tabular}\end{table}",
            r"\begin{table}[t]\centering\caption{Median cost per retraining round (excluding initial verification), with ranges in the reproduction. Oracle time includes bound computation; evaluation counts include screens and enclosure operations.}\label{tab:timings}",
            r"\small\begin{tabular}{lrrr}\hline Study & Learner (s) & Oracle (s) & Evaluations\\\hline"]
    for study,x in r["timings"].items():
        tex.append(study+" & "+" & ".join("$"+f(x[k]["median"],sig=3)+"$" for k in
                                         ("train_s", "verify_including_bounds_s", "oracle_evaluations"))+r"\\")
    hardware = r["machine"]["cpu_model"].replace("_",r"\_")
    values=[value for value in r["machine"]["thread_environment"].values() if value is not None]
    single=bool(values) and all(value=="1" for value in values)
    thread_note="one BLAS thread" if single else "BLAS settings in the manifest"
    tex.append(r"\hline\multicolumn{4}{l}{\footnotesize "+hardware+"; Python "+r["machine"]["python"]+"; "+thread_note+r".}\\")
    tex.append(r"\multicolumn{4}{l}{\footnotesize Independent producers may overlap; timings include contention.}\\")
    tex += [r"\hline\end{tabular}\end{table}"]
    tex += [r"\begin{table}[t]\centering\caption{Budgeted grid refinement of frozen candidates. The last two columns compare valid margin bounds on the final grid; the coordinatewise bound does not change the learner.}\label{tab:refinement}",
            r"\small\begin{tabular}{rrrrr}\hline Seed & Final $N_{\rm ax}$ & Checks & Euclidean margin & Coordinate margin\\\hline"]
    for record in r["refinement"]["records"]:
        last=record["steps"][-1]
        values=[f(record["seed"]),f(last["axis_points"]),f(len(record["steps"])),
                fu(mul_up(last["L"],last["tau"]),sig=3),
                fu(record["coordinate_bound_at_final_grid"]["coordinate_margin_upper"],sig=3)]
        tex.append(" & ".join("$"+value+"$" for value in values)+r"\\")
    tex += [r"\hline\end{tabular}\end{table}"]
    # Added below from the declared revision-experiment schemas.
    tex.extend(render_new_results(load("revision_experiments.json"), load("revision_extensions.json")))
    (PAPER / "revision_tables.tex").write_text("\n".join(tex)+"\n")


def render_new_results(experiments, extensions):
    """Render results without choosing favorable subsets after observing outcomes."""
    import fill_tex
    f, fu, fd=fill_tex.fmt,fill_tex.fmt_up,fill_tex.fmt_down
    if experiments.get("status") != "pass" or experiments.get("pilot"):
        raise ValueError("production ablation results are incomplete or a pilot")
    if extensions["config"].get("pilot") or extensions["config"]["n_seeds"] != 20:
        raise ValueError("production extension results must contain all declared seeds")
    tex=[r"\begin{table}[t]\centering\caption{Matched operating-grid ablation and constrained controls. All operating-grid regimes share seeds and budgets. Bounded comparisons cover coupled margins only. The final row uses the coarse grid.}\label{tab:ablation}",
         r"\small\begin{tabular}{lrrrrr}\hline Test / rule & $N_{\rm ax}$ & Cert. & Div. & Stall & Fail\\\hline"]
    names={"raw_fixed":"Raw / fixed", "raw_coupled":"Raw / coupled",
           "normalized_fixed":"Normalized / fixed", "normalized_coupled":"Normalized / coupled",
           "raw_coupled_bounded":"Raw / bounded coupled",
           "normalized_coupled_bounded":"Normalized / bounded coupled",
           "normalized_coupled_bounded_coarse":"Normalized / bounded coarse"}
    for name,label in names.items():
        s=experiments["summary"][name]
        runs=experiments["regimes"][name]
        first=next(iter(runs.values()))
        axis=first["protocol"]["n_grid"]
        tex.append(label+" & "+" & ".join(str(v) for v in (axis,s["certified"],s["diverged"],s["stalled"],s["numerical_failure"]))+r"\\")
    tex += [r"\hline\end{tabular}\end{table}",
            r"\begin{table}[t]\centering\caption{Short-horizon sensitivity on a common coarse diagnostic grid. These counts are not operating-grid success rates. The pendulum row is a separate matched fixed-margin control at its original operating resolution.}\label{tab:sensitivity}",
            r"\small\begin{tabular}{lrrrr}\hline Setting & Seeds & Cert. & Div. & Stall\\\hline"]
    for name,s in experiments["sensitivity"]["summary"].items():
        label=name.replace("sensitivity_gamma_",r"$\gamma=").replace("_k_",r"$, $q_c=")+"$"
        tex.append(label+" & "+" & ".join(str(s[k]) for k in ("n","certified","diverged","stalled"))+r"\\")
    s=experiments["pendulum_fixed"]["summary"]
    tex.append("Pendulum / fixed & "+" & ".join(str(s[k]) for k in ("n","certified","diverged","stalled"))+r"\\")
    tex += [r"\hline\end{tabular}\end{table}",
            r"\begin{table}[t]\centering\caption{Adaptive verification and fitted-surrogate transfer. Box and grid-operation counts have different units. The surrogate residual bounds use known polynomial truth over the entire box.}\label{tab:extensions}",
            r"\small\begin{tabular}{lr}\hline Quantity & Result\\\hline"]
    three=extensions["three_dimensional"]
    tex.append("3D certified / seeds & "+str(three["certified_count"])+" / "+str(extensions["config"]["n_seeds"])+r"\\")
    tex.append("3D inconclusive & "+str(three["uncertified_count"])+r"\\")
    if three["certified_rounds"]:
        tex.append("3D median retraining rounds & $"+f(float(np.median(three["certified_rounds"])))+r"$\\")
        counts=[run["final_oracle"]["counters"]["boxes_evaluated"] for run in three["runs"] if run["certified"]]
        tex.append("3D median boxes (certified runs) & $"+f(float(np.median(counts)))+r"$\\")
        representative=min((run for run in three["runs"] if run["certified"]),key=lambda run:run["seed"])
        guaranteed=representative["guarantees"]
        tex.append("3D representative seed & $"+f(representative["seed"])+r"$\\")
        for label,key,formatter in [(r"3D $a$","a",fd),(r"3D $b$","b_upper",fu),
                                    (r"3D raw $\eta$","eta_lower",fd),(r"3D rate $\lambda'$","rate_lower",fd),
                                    (r"3D overshoot","overshoot_upper",fu)]:
            tex.append(label+" & $"+formatter(guaranteed[key],sig=8 if key=="rate_lower" else 4)+r"$\\")
    for record in extensions["frozen_oracle_comparison"]["records"]:
        label="Frozen seed "+str(record["seed"])
        tex.append(label+" adaptive boxes & $"+f(record["adaptive"]["counters"]["boxes_evaluated"])+r"$\\")
        tex.append(label+" grid / adaptive time (s) & $"+f(record["grid"]["wall_s"],sig=3)+" / "+f(record["adaptive"]["wall_s"],sig=3)+r"$\\")
    surrogate=extensions["learned_surrogate"]
    tex.append("Surrogate trajectory samples & $"+f(surrogate["fitting"]["sample_count"])+r"$\\")
    tex.append("Surrogate derivative-fit RMSE & $"+f(surrogate["fitting"]["training_rmse"],sig=3)+r"$\\")
    transfer=surrogate["transfer"]
    if transfer is None:
        tex.append(r"Surrogate transfer & Inconclusive\\")
    else:
        for label,key,formatter in [(r"Surrogate $\eta$","eta_lower",fd),
                                   (r"Metric $b$","b_upper",fu), (r"Metric $\mu_1$","mu1_upper",fu),
                                   (r"Residual $\delta_0$","delta0_upper",fu),
                                   (r"Residual $\delta_1$","delta1_upper",fu),
                                   (r"Remaining raw margin","remaining_margin_lower",fd),
                                   (r"True-system rate $\lambda''$","true_rate_lower",fd),
                                   (r"Overshoot $\sqrt{b/a}$","overshoot_upper",fu)]:
            tex.append(label+" & $"+formatter(transfer[key],sig=8 if key=="true_rate_lower" else 4)+r"$\\")
        tex.append("Transfer certified & "+("Yes" if transfer["certified"] else "No")+r"\\")
    tex += [r"\hline\end{tabular}\end{table}"]
    return tex


if __name__ == "__main__":
    result=generate()
    print("Generated complete certificate reporting:",len(result["certificates"]),"certified candidates")
