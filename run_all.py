"""Reproduce and validate every declared stage, stopping at the first failure.

Run ``python3 run_all.py`` from any directory. Complete output is captured in
results/run_all.log and results/stage_logs/. results/run_manifest.json records
the environment, source/artifact hashes, stage outcomes, and elapsed times.
Successful process exit alone is insufficient: outputs must be fresh and pass
scientific consistency checks. No empirical bound enters a certificate.
"""
import contextlib
import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
PAPER = HERE / "paper"
RESULTS = HERE / "results"
TEX = "nsp_contraction_paper_ieee"
PRECOMPUTED = {}
SCIENCE_REPLAY_WORKERS = 1


class ValidationError(RuntimeError):
    """A completed stage did not produce evidence sufficient for PASS."""


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes():
    paths = list(HERE.glob("*.py")) + list(HERE.glob("requirements*.txt"))
    paths += list((HERE / "tests").glob("*.py"))
    paths += list((HERE / "release").glob("*.py"))
    paths += list((HERE / "reviews").glob("*.md"))
    paths += [PAPER / (TEX + ".tex"), HERE / "README.md", HERE / "CLAUDE.md",
              HERE / "RESPONSE_TO_REFEREE.md", HERE / "RESPONSE_TO_CLAUDE.md", HERE / "CITATION.cff"]
    return {str(p.relative_to(HERE)): sha256(p) for p in sorted(paths) if p.is_file()}


def artifact_hashes():
    paths = [RESULTS / name for name in (
        "traces.json", "gridcost.json", "residual.json", "pendulum.json",
        "neural.json", "smoke_p1.json", "timing_polynomial.json",
        "timing_pendulum.json", "timing_neural.json", "rounding_checks.json",
        "revision_experiments.json", "revision_extensions.json", "revision_report.json",
        "timing_revision.json", "arxiv_bundle.json")]
    paths += [PAPER / "numbers.tex", PAPER / "revision_numbers.tex",
              PAPER / "revision_tables.tex", PAPER / (TEX + ".pdf")]
    paths += list((HERE / "dist").glob("*-claude-revision.tar.gz"))
    paths += list(PAPER.glob("fig_*.pdf")) + list(PAPER.glob("fig_*.png"))
    return {str(p.relative_to(HERE)): sha256(p) for p in sorted(paths) if p.is_file()}


def _capture(command):
    try:
        p = subprocess.run(command, cwd=HERE, capture_output=True, text=True, check=False)
        return p.stdout.strip() if p.returncode == 0 else None
    except OSError:
        return None


def environment():
    packages = {}
    for name in ("numpy", "autograd", "matplotlib", "gmpy2"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    blas = io.StringIO()
    try:
        import numpy as np
        with contextlib.redirect_stdout(blas):
            np.show_config()
    except ImportError:
        blas.write("numpy not installed")
    latex_version = _capture(["pdflatex", "--version"])
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "installed_packages": dict(sorted(
            (dist.metadata["Name"], dist.version)
            for dist in importlib.metadata.distributions() if dist.metadata.get("Name"))),
        "numpy_configuration": blas.getvalue(),
        "thread_environment": {key: os.environ.get(key) for key in (
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "pdflatex": latex_version.splitlines()[0] if latex_version else None,
        "git_commit": _capture(["git", "rev-parse", "HEAD"]),
        "git_status": _capture(["git", "status", "--short"]),
    }


def require(condition, detail):
    if not condition:
        raise ValidationError(detail)


def finite(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"{name} must be a finite number")
    return float(value)


def require_outward(data):
    backend = data.get("verification_backend", {})
    require(isinstance(backend, dict) and backend.get("outward_rounded") is True,
            "missing outward-rounded verification backend")


def validate_grid_certificate(data, L_key="L", worst_key="worst", slack_key="uniform_slack"):
    from rounding import mul_up, sub_down
    require(data.get("grid_enclosed") is True and data.get("outward_rounded") is True,
            "positive result did not enclose the grid with outward rounding")
    L = finite(data[L_key], L_key)
    tau = finite(data["tau"], "covering radius")
    worst = finite(data[worst_key], worst_key)
    slack = finite(data[slack_key], slack_key)
    require(L >= 0 and tau >= 0, "invalid certificate constants")
    lower = sub_down(-worst, mul_up(L, tau))
    require(0 <= slack <= lower,
            "reported slack is not a downward-rounded nonnegative guarantee")
    return slack


def validate_protocol(data):
    config = data.get("config", {})
    require(isinstance(config, dict), "missing experiment protocol")
    for key in ("max_rounds", "training_steps", "initial_samples"):
        require(isinstance(config.get(key), int) and not isinstance(config[key], bool)
                and config[key] > 0, f"protocol {key} must be a positive integer")
    for key in ("growth_factor", "learning_rate", "base_margin", "gamma"):
        require(finite(config.get(key), f"protocol {key}") > 0,
                f"protocol {key} must be positive")
    require(config["growth_factor"] > 1 and config["gamma"] > 1,
            "protocol growth factor and coupling must exceed one")
    require(config.get("optimizer") == "Adam"
            and config.get("optimizer_state") == "reset each retraining round",
            "missing optimizer protocol")
    criterion = config.get("outcome_criterion", {})
    require(isinstance(criterion, dict)
            and all(isinstance(criterion.get(key), str) and criterion[key]
                    for key in ("certified", "diverged", "stalled")),
            "missing operational outcome definitions")
    return config


def validate_outcome_protocol(run, config):
    trace = run["trace"]
    require(0 < len(trace) <= config["max_rounds"] + 1,
            "trace exceeds recorded round cap")
    constants = [finite(row.get("L"), "trace Lipschitz constant") for row in trace]
    require(all(value >= 0 for value in constants), "negative trace Lipschitz constant")
    diverged = not run["certified"] and max(constants) > config["growth_factor"] * constants[0]
    require(run["diverged"] == diverged,
            "outcome disagrees with recorded Lipschitz-growth criterion")
    if "stalled" in run:
        require(run["stalled"] == (not run["certified"] and not diverged),
                "stalled outcome disagrees with recorded criterion")


def validate_neural_metric_outputs(data, slack, config, depth):
    from fractions import Fraction
    require(data.get("metric_bounds_outward_rounded") is True
            and data.get("metric_bounds_depth") == depth,
            "missing outward global neural metric bounds")
    a, b = finite(data.get("a"), "neural a"), finite(data.get("b"), "neural b")
    require(0 < a <= b and a == finite(config.get("metric_floor"), "metric floor"),
            "invalid neural metric bounds")
    rate0 = finite(config.get("target_rate"), "target rate")
    eta = finite(data.get("raw_margin_lower"), "raw margin")
    rate = finite(data.get("rate_lower"), "rate lower bound")
    overshoot = finite(data.get("overshoot_upper"), "overshoot upper bound")
    F = Fraction.from_float
    require(0 <= F(eta) <= 2 * F(a) * F(slack), "unsafe neural raw-margin conversion")
    require(F(rate0) <= F(rate) <= F(rate0) + F(eta) / (2 * F(b)),
            "unsafe neural rate bound")
    require(overshoot > 0 and F(overshoot) ** 2 >= F(b) / F(a),
            "unsafe neural overshoot bound")


def read_json(name):
    def reject_constant(value):
        raise ValidationError(f"{name} contains {value}")
    with (RESULTS / name).open() as stream:
        data = json.load(stream, parse_constant=reject_constant)
    require(isinstance(data, dict), f"{name} must contain an object")
    def check_finite(value, location):
        if isinstance(value, float):
            require(math.isfinite(value), f"{location} is not finite")
        elif isinstance(value, dict):
            for key, item in value.items():
                check_finite(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                check_finite(item, f"{location}[{index}]")
    check_finite(data, name)
    return data


def require_fresh(paths, started_ns):
    for path in paths:
        path = Path(path)
        require(path.is_file(), f"missing output: {path}")
        require(path.stat().st_size > 0, f"empty output: {path}")
        require(path.stat().st_mtime_ns >= started_ns,
                f"stale output (not regenerated by this stage): {path}")


def validate_traces(data):
    require_outward(data)
    config = validate_protocol(data)
    n = data["config"]["n_seeds"]
    require(isinstance(n, int) and n > 0, "main experiment has no seeds")
    for regime in ("stall", "diverge", "certify"):
        summary = data["summary"][regime]
        counts = [summary[key] for key in ("certified", "diverged", "stalled")]
        require(all(isinstance(v, int) and v >= 0 for v in counts)
                and sum(counts) == n, f"invalid outcome counts for {regime}")
        runs = data["regimes"][regime]
        require(len(runs) == n, f"missing seed runs for {regime}")
        for key, count in zip(("certified", "diverged", "stalled"), counts):
            require(sum(run[key] is True for run in runs.values()) == count,
                    f"{regime} {key} count disagrees with seed outcomes")
        for run in runs.values():
            require(all(isinstance(run[key], bool) for key in ("certified", "diverged", "stalled"))
                    and sum(run[key] for key in ("certified", "diverged", "stalled")) == 1,
                    f"{regime} seed outcome flags do not form a partition")
            validate_outcome_protocol(run, config)
            final_gap = finite(run["trace"][-1]["gap"], "final certifiability gap")
            require(run["certified"] == (final_gap <= 0),
                    f"{regime} certification flag disagrees with final margin")
            if run["certified"]:
                validate_grid_certificate(run["trace"][-1])
    require(data["summary"]["certify"]["certified"] > 0,
            "operating-grid experiment produced no certificate")


def validate_residual(data):
    require_outward(data)
    require(data.get("grid_enclosed") is True, "residual margin did not enclose the grid")
    require(data.get("budget_outward_rounded") is True,
            "residual budget did not use directed arithmetic")
    for key in ("eta", "b_upper", "delta_star"):
        require(finite(data[key], key) > 0, f"residual {key} must be positive")
    rows = data["rows"]
    require(bool(rows), "residual sweep is empty")
    for row in rows:
        measured = finite(row["measured"], "measured residual condition")
        bound = finite(row["bound"], "residual bound")
        require(measured <= bound + 1e-9, "residual measurement exceeds theorem bound")


def validate_benchmark(data, neural=False):
    require_outward(data)
    config = validate_protocol(data)
    n = data["n_seeds"]
    cert_key, div_key = (("certified_op", "diverged_op") if neural
                         else ("certified", "diverged"))
    cert, div = data[cert_key], data[div_key]
    require(isinstance(n, int) and n > 0, "benchmark has no seeds")
    require(isinstance(cert, int) and isinstance(div, int)
            and cert > 0 and div >= 0 and cert + div <= n,
            "benchmark has invalid counts or no certified candidate")
    validation = data.get("validation")
    require(isinstance(validation, dict) and bool(validation)
            and all(value is True for value in validation.values()),
            "benchmark acceptance checks are missing or failed")
    example = data["example"]
    require(example.get("certified") is True, "benchmark example is not certified")
    L = finite(example["L_cert"], "L_cert")
    slack = finite(example["slack"], "slack")
    tau = finite(example["tau"], "tau")
    require(L >= 0 and tau >= 0 and slack - L * tau >= 0,
            "benchmark example fails its Lipschitz margin")
    uniform = validate_grid_certificate(dict(example, worst=-slack), L_key="L_cert")
    runs = data["runs"]
    require(len(runs) == n, "benchmark is missing seed runs")
    require(sum(run["certified"] is True for run in runs.values()) == cert
            and sum(run["diverged"] is True for run in runs.values()) == div,
            "benchmark summary disagrees with seed outcomes")
    for seed, run in runs.items():
        require(isinstance(run["certified"], bool) and isinstance(run["diverged"], bool)
                and not (run["certified"] and run["diverged"]),
                f"benchmark seed {seed} has inconsistent outcome flags")
        require(bool(run["trace"]), f"benchmark seed {seed} has no trace")
        validate_outcome_protocol(run, config)
        final_gap = finite(run["trace"][-1]["gap"], f"benchmark seed {seed} final gap")
        require(run["certified"] == (final_gap <= 0),
                f"benchmark seed {seed} certification flag disagrees with final margin")
        if run["certified"]:
            validate_grid_certificate(run["trace"][-1])
    if neural:
        validate_neural_metric_outputs(example, uniform, config, data["depth"])
        grid_points = data["op_grid"] ** 2
        require(example.get("gridpoints_covered") == grid_points,
                "neural example does not cover every operating-grid node")
        for seed, run in runs.items():
            if run["certified"]:
                validate_neural_metric_outputs(run, run["trace"][-1]["uniform_slack"],
                                              config, data["depth"])
                require(run["trace"][-1].get("gridpoints_covered") == grid_points,
                        f"neural seed {seed} has an incomplete grid proof")
        coarse = data["certified_coarse"]
        require(isinstance(coarse, int) and 0 <= coarse <= cert,
                "invalid neural coarse-grid certification count")
        require(all(isinstance(run["certified_coarse"], bool)
                    and (not run["certified_coarse"] or run["certified"])
                    for run in runs.values())
                and sum(run["certified_coarse"] for run in runs.values()) == coarse,
                "neural coarse-grid count disagrees with seed outcomes")
        for seed, run in runs.items():
            if run["certified_coarse"]:
                require(run.get("coarse_grid_enclosed") is True
                        and run.get("coarse_gridpoints_covered") == data["coarse_grid"] ** 2
                        and finite(run.get("coarse_uniform_slack"), "coarse uniform slack") >= 0,
                        f"neural seed {seed} has an incomplete coarse-grid proof")
    return f"certified {cert}/{n}; diverged {div}; example uniform slack {uniform:.6g}"


def validate_smoke(data):
    require_outward(data)
    require(data["candidate"]["certified"] is True, "smoke candidate did not certify")
    s = finite(data["s"], "smoke uniform slack")
    L = finite(data["L_cert"], "smoke L_cert")
    tau = finite(data["tau"], "smoke tau")
    worst = finite(data["worst_gt"], "smoke worst_gt")
    require(L >= 0 and tau >= 0 and s >= 0, "smoke uniform slack is negative")
    validate_grid_certificate(dict(data, grid_enclosed=data["candidate"].get("grid_enclosed"),
                                   outward_rounded=data["candidate"].get("outward_rounded")),
                              L_key="L_cert", worst_key="worst_gt", slack_key="s")
    return f"certified smoke candidate; uniform slack {s:.6g}"


def run_command(command, cwd=HERE):
    print("$ " + " ".join(map(str, command)), flush=True)
    output = []
    with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                output.append(line)
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    require(code == 0, f"process exited {code}: {' '.join(map(str, command))}")
    return "".join(output)


def run_script(script, args=(), outputs=(), validator=None):
    prepared=PRECOMPUTED.get(script)
    if prepared is None:
        started_ns = time.time_ns()
        run_command([sys.executable, "-u", str(HERE / script), *args])
    else:
        if tuple(args) != prepared["args"]:
            raise ValidationError("prefetched producer arguments changed")
        PRECOMPUTED.pop(script)
        started_ns=prepared["started_ns"]
        process=prepared["process"]
        try:
            code=process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise
        finally:
            prepared["stream"].close()
        print(f"$ {script} (independent producer started earlier; full output follows)",flush=True)
        print(prepared["path"].read_text(errors="replace"),end="",flush=True)
        require(code==0,f"prefetched process exited {code}: {script}")
    require_fresh([HERE / p for p in outputs], started_ns)
    for path in outputs:
        if str(path).endswith(".json"):
            read_json(Path(path).name)
    return validator() if validator else "fresh outputs validated"


def start_prefetch(script,args=()):
    """Run an independent producer once, preserving exit status and fresh-output epoch."""
    if script in PRECOMPUTED:
        raise ValidationError("producer already running: "+script)
    path=RESULTS/"stage_logs"/("prefetch_"+Path(script).stem+".log")
    stream=path.open("w",buffering=1)
    started_ns=time.time_ns()
    try:
        process=subprocess.Popen([sys.executable,"-u",str(HERE/script),*args],cwd=HERE,
                                 stdout=stream,stderr=subprocess.STDOUT,text=True)
    except BaseException:
        stream.close()
        raise
    PRECOMPUTED[script]=dict(process=process,stream=stream,path=path,
                             started_ns=started_ns,args=tuple(args))
    print("Started independent scientific producer: "+script,flush=True)


def stop_prefetch():
    for entry in PRECOMPUTED.values():
        if entry["process"].poll() is None:
            entry["process"].terminate()
            entry["process"].wait()
        entry["stream"].close()
    PRECOMPUTED.clear()


def bounds_tests():
    import bounds
    import rounding
    require(rounding.self_test() is True, "directed arithmetic acceptance failed")
    run_command([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    run_command([sys.executable, "-m", "unittest", "discover", "-s", "release", "-p", "test_*.py", "-v"])
    checks = {name: bool(getattr(bounds, name)(verbose=True)) for name in (
        "test_1b", "test_1c_lower_bound", "test_1c_monotone")}
    require(all(checks.values()), f"bounds acceptance checks failed: {checks}")
    evidence = {"passed": True, "verification_backend": rounding.backend_info(),
                "checks": checks, "regression_tests": "unittest discovery passed",
                "checked_at": utc_now()}
    (RESULTS / "rounding_checks.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return "directed arithmetic, regression tests, and polynomial acceptance checks passed"


def validate_latex_log(log):
    bad = re.findall(
        r"Undefined control sequence|LaTeX Error|Emergency stop|Fatal error|"
        r"(?:Reference|Citation)[^\n]*undefined|There were undefined (?:references|citations)|"
        r"multiply[- ]defined|Label[^\n]*multiply defined", log, flags=re.IGNORECASE)
    require(not bad, "LaTeX errors or unresolved references/citations: " + ", ".join(bad))
    match = re.search(r"Output written on .*\((\d+) pages", log)
    require(match is not None, "LaTeX did not report a completed PDF")
    warnings = re.findall(r"^.*(?:Warning:|Overfull \\[hv]box|Underfull \\[hv]box).*$",
                          log, flags=re.MULTILINE)
    return int(match.group(1)), warnings


def latex():
    started_ns = time.time_ns()
    for attempt in range(1,5):
        run_command(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TEX], cwd=PAPER)
        if attempt>=2:
            current=(PAPER/(TEX+".log")).read_text(errors="replace")
            if not re.search(r"Label\(s\) may have changed|rerunfilecheck Warning|Rerun to get",current):
                break
    log_path = PAPER / (TEX + ".log")
    pdf_path = PAPER / (TEX + ".pdf")
    require_fresh([log_path, pdf_path], started_ns)
    require(pdf_path.read_bytes().startswith(b"%PDF-"), "LaTeX output is not a PDF")
    pages, warnings = validate_latex_log(log_path.read_text(errors="replace"))
    for warning in warnings:
        print("LaTeX warning: " + warning)
    return f"{pages} pages, zero errors/undefined references/citations; {len(warnings)} warning(s)"


def validate_revision_experiments():
    import revision_experiments
    data = read_json("revision_experiments.json")
    require_outward(data)
    revision_experiments.validate_results(data)
    return "controlled ablation, constrained variants, scale illustration and controls validated"


def validate_revision_extensions():
    import revision_extensions
    data = read_json("revision_extensions.json")
    require_outward(data)
    revision_extensions.validate_results(data,replay_workers=SCIENCE_REPLAY_WORKERS)
    return "adaptive proof coverage, 3D synthesis and fitted-model transfer validated"


def validate_arxiv_bundle():
    data = read_json("arxiv_bundle.json")
    require(data.get("status") == "pass" and data.get("clean_build", {}).get("status") == "pass",
            "arXiv source bundle did not compile cleanly")
    require(data["clean_build"]["warnings"] == [], "clean arXiv build has layout or reference warnings")
    require(sha256(HERE / data["archive"]) == data["sha256"], "arXiv archive hash mismatch")
    return f"{data['member_count']} source files; clean {data['clean_build']['passes']}-pass build, zero warnings"


def stages():
    return [
        ("bounds self-tests", bounds_tests),
        ("multi-seed regimes", lambda: run_script("experiments.py", outputs=(
            "results/traces.json", "results/gridcost.json", "results/timing_polynomial.json", "paper/fig_traces.pdf",
            "paper/fig_heatmaps.pdf", "paper/fig_gridcost.pdf"),
            validator=lambda: (validate_traces(read_json("traces.json")) or "main outcomes validated"))),
        ("residual experiment", lambda: run_script("residual.py", outputs=(
            "results/residual.json", "paper/fig_residual.pdf"),
            validator=lambda: (validate_residual(read_json("residual.json")) or "residual bounds validated"))),
        ("non-polynomial benchmark", lambda: run_script("bench_pendulum.py", ("--experiment",), outputs=(
            "results/pendulum.json", "results/timing_pendulum.json", "paper/fig_pendulum.pdf"),
            validator=lambda: validate_benchmark(read_json("pendulum.json")))),
        ("neural-metric benchmark", lambda: run_script("bench_neural.py", ("--experiment",), outputs=(
            "results/neural.json", "results/timing_neural.json", "paper/fig_neural.pdf"),
            validator=lambda: validate_benchmark(read_json("neural.json"), neural=True))),
        ("P1 smoke diagnostic", lambda: run_script("smoke_p1.py", outputs=("results/smoke_p1.json",),
            validator=lambda: validate_smoke(read_json("smoke_p1.json")))),
        ("controlled revision experiments", lambda: run_script("revision_experiments.py", outputs=(
            "results/revision_experiments.json", "results/timing_revision.json",
            "paper/fig_ablation.pdf", "paper/fig_scale_chase.pdf"),
            validator=validate_revision_experiments)),
        ("adaptive higher-dimensional and surrogate extensions", lambda: run_script("revision_extensions.py", outputs=(
            "results/revision_extensions.json", "paper/fig_extension3d.pdf", "paper/fig_surrogate.pdf"),
            validator=validate_revision_extensions)),
        ("complete certified reporting", lambda: run_script("revision_report.py", outputs=(
            "results/revision_report.json",))),
        ("fill_tex", lambda: run_script("fill_tex.py", outputs=("paper/numbers.tex",
            "paper/revision_numbers.tex", "paper/revision_tables.tex"))),
        ("pdflatex stable references + log check", latex),
        ("clean arXiv source bundle", lambda: run_script("release/build_arxiv.py", ("--check",), outputs=(
            "results/arxiv_bundle.json", "dist/neurosymbolic-arxiv-claude-revision.tar.gz"),
            validator=validate_arxiv_bundle)),
    ]


def main(parallel_science=False):
    global SCIENCE_REPLAY_WORKERS
    SCIENCE_REPLAY_WORKERS=4 if parallel_science else 1
    pipeline_started = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    log_dir = RESULTS / "stage_logs"
    log_dir.mkdir(exist_ok=True)
    declared_stages = stages()
    manifest = {"schema_version": 2, "started_at": utc_now(), "status": "running",
                "expected_stage_count": len(declared_stages),
                "execution_mode": "overlapping independent scientific producers" if parallel_science else "sequential",
                "environment": environment(), "sources_at_start": source_hashes(), "stages": []}
    manifest_path = RESULTS / "run_manifest.json"

    def save_manifest():
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(manifest_path)

    all_ok = True
    with (RESULTS / "run_all.log").open("w", buffering=1) as full_log:
        with contextlib.redirect_stdout(Tee(sys.stdout, full_log)), contextlib.redirect_stderr(Tee(sys.stderr, full_log)):
            print(f"Reproduction started {manifest['started_at']}")
            save_manifest()
            try:
                for index, (name, function) in enumerate(declared_stages, 1):
                    stage_path = log_dir / f"{index:02d}_{name.replace(' ', '_')}.log"
                    item = {"index": index, "name": name, "started_at": utc_now(),
                            "source_sha256": source_hashes(), "status": "running"}
                    manifest["stages"].append(item)
                    save_manifest()
                    started = time.perf_counter()
                    with stage_path.open("w", buffering=1) as stage_log:
                        with contextlib.redirect_stdout(Tee(sys.stdout, stage_log)), contextlib.redirect_stderr(Tee(sys.stderr, stage_log)):
                            print(f"\n[{index}/{len(declared_stages)}] {name}")
                            try:
                                detail = function()
                                item.update(status="pass", detail=detail)
                            except Exception as error:
                                item.update(status="fail", detail=f"{type(error).__name__}: {error}")
                                all_ok = False
                            item.update(finished_at=utc_now(), elapsed_s=time.perf_counter()-started)
                            print(f"  -> {item['status'].upper()}: {item['detail']} ({item['elapsed_s']:.1f}s)")
                    item.update(log=str(stage_path.relative_to(HERE)), log_sha256=sha256(stage_path))
                    save_manifest()
                    if not all_ok:
                        print("Pipeline stopped; later stages were not run.")
                        break
                    if parallel_science and name=="bounds self-tests":
                        start_prefetch("bench_neural.py",("--experiment",))
                        start_prefetch("revision_experiments.py")
                    if parallel_science and name=="multi-seed regimes":
                        start_prefetch("revision_extensions.py")
            except KeyboardInterrupt:
                all_ok = False
                manifest["status"] = "interrupted"
                if manifest["stages"] and manifest["stages"][-1]["status"] == "running":
                    manifest["stages"][-1].update(status="interrupted", finished_at=utc_now())
                print("Pipeline interrupted; later stages were not run.")
            except Exception as error:
                all_ok=False
                manifest["runner_error"]=f"{type(error).__name__}: {error}"
                print("Pipeline orchestration failed: "+manifest["runner_error"])
            finally:
                stop_prefetch()
            if manifest["status"] != "interrupted":
                manifest["status"] = "pass" if all_ok else "fail"
            manifest.update(finished_at=utc_now(), sources_at_finish=source_hashes(),
                            elapsed_s=time.perf_counter()-pipeline_started,
                            artifact_sha256=artifact_hashes())
            save_manifest()
            print("\nSUMMARY")
            for item in manifest["stages"]:
                print(f"  {item['status'].upper():11} {item['index']}. {item['name']}: {item.get('detail', '')}")
            print("\n" + ("ALL STAGES PASS" if all_ok else "PIPELINE INCOMPLETE OR FAILED"))
    manifest.update(run_log="results/run_all.log", run_log_sha256=sha256(RESULTS / "run_all.log"))
    save_manifest()
    return 0 if all_ok else 1


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel-science",action="store_true",
                        help="overlap independent scientific producers; per-study timings include contention")
    sys.exit(main(parallel_science=parser.parse_args().parallel_science))
