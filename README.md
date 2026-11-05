# Counterexample-Guided Synthesis of Contraction Metrics with Certified Verification

This working revision addresses the November 2026 external referee report.
It is developed on `codex/claude-revision` from immutable release v1.0.1.
The new experiments and manuscript are local revision artifacts, not a new
published release. `RESPONSE_TO_CLAUDE.md` records every review disposition;
`reviews/claude-2026-11-04.md` preserves the source report as review evidence.

Research code and manuscript for CEGIS-C: counterexample-guided learning and
verification of continuous-time autonomous contraction metrics. The paper
analyzes scale degeneracy, fixed-margin stalling, and the conversion of verified
slack into rate and model-error guarantees.

## Reproduce the paper

The reference Python version is **3.12.7**. Install the runtime dependencies in
an isolated Python 3.12 environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

`requirements.txt` pins the direct dependencies, including gmpy2 for MPFR
directed arithmetic; `requirements-lock.txt`
also pins their runtime dependencies to the local reproduction environment.
Install a TeX distribution providing `pdflatex`, `IEEEtran.cls`, `amsmath`,
`amssymb`, `amsthm`, `etoolbox`, `algorithm`, `algpseudocode`, `graphicx`,
`hyperref`, `xcolor`, `placeins`, and `lineno`. PDF inspection additionally uses Poppler's `pdfinfo`
and `pdftoppm`; these are not needed by the experiment scripts.

From the repository root, run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python run_all.py
```

The runner executes twelve stages: arithmetic and regression checks,
three-regime multi-seed experiments, residual robustness, pendulum benchmark,
neural metric benchmark, smoke verification, controlled revision experiments,
adaptive verification and higher-dimensional/learned-surrogate extensions,
complete certificate reporting, LaTeX-number generation, two LaTeX passes,
and a clean arXiv source-bundle build. It returns a nonzero exit status on failure. The neural study
dominates runtime; it evaluates grids with up to 8001 points per axis over 20
seeds. It retains only the worst counterexamples between evaluation blocks.
The reference command fixes BLAS threading to one thread for predictable
execution and to avoid overhead on these small matrix products. The runner
records these settings; it also works with inherited settings, whose numerical
paths and timings can differ.

The rebuilt paper is `paper/nsp_contraction_paper_ieee.pdf`. The runner writes
`results/run_all.log`, individual stage logs, and `results/run_manifest.json`.
The manifest records the environment, source hashes, artifact hashes, and
stage outcomes and timings. A Git revision alone does not identify a dirty
working tree; use the recorded file hashes to identify the actual inputs.
Do not redirect output to `results/run_all.log`: the runner manages that file.
Option `--parallel-science` overlaps the independent neural, ablation, and
extension producers after their prerequisites. It keeps separate producer
logs, checks each exit status and freshness epoch, and stops remaining workers
on failure. Timings in this mode include competition for machine resources.
The paper uses a single-column, single-spaced preprint layout with continuous
line numbers, including displayed equations, for feedback. It retains the
standard IEEEtran class and omits its double-spaced referee-mode option.

After a passing run, create the complete local reviewer archive:

```sh
python package_reproduction.py
```

This writes `dist/neurosymbolic-reproduction.zip` and its SHA256 checksum.
The archive contains the source, locked dependency specifications, manuscript, figures,
scientific results, stage logs, and run manifest. Its `SNAPSHOT.json` records
the hash of every included file. Packaging rejects changed inputs or outputs,
failed or incomplete runs, and source changes during the run. Repackaging the
same passing snapshot produces the same ZIP bytes.

## Published snapshot and final manuscript

[Release v1.0.1](https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.0.1)
provides the exact verified reproduction ZIP and its checksum, together with
the updated line-numbered PDF, TeX source, generated numbers, a tested arXiv
source bundle, and editorial-build provenance. GitHub release immutability
protects the published tag and assets. The earlier
[v1.0.0 release](https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.0.0)
remains unchanged.
The verified ZIP has SHA256
`c67b4247793c591125dfd71c1d86776a21b95ba1a59d116126c51d29cd934070`.

The ZIP preserves the complete passing scientific run, including the manuscript
at that run. The separately attached manuscript includes the final wording,
the Fitzsimmons--Liu related-work comparison, and the current release citation.
The tag's paper sources contain these edits. `results/editorial_rebuild.json`
records the parent snapshot, unchanged
scientific inputs and outputs, the editorial changes, and the new PDF build.
The original `results/run_manifest.json` and scientific results remain
unchanged; the editorial rebuild does not claim another full experimental run.

To rebuild only the final manuscript from the tag's sources and saved results:

```sh
python fill_tex.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error nsp_contraction_paper_ieee
pdflatex -interaction=nonstopmode -halt-on-error nsp_contraction_paper_ieee
```

Use the full `run_all.py` command above for a fresh scientific reproduction.
The strict packager intentionally requires that fresh passing run before
packaging a source tree containing later editorial changes. Release asset
`SHA256SUMS` lists the published checksums.

## arXiv source bundle

The baseline release asset `neurosymbolic-arxiv-v1.0.1.tar.gz` contains exactly the
top-level manuscript TeX, `numbers.tex`, and the six included figure PDFs.
It has no compiled manuscript, logs, experiment data, or nested archive.
The bibliography is embedded in the TeX, so no separate BibTeX input is needed.
The bundle preserves the single-spaced manuscript and continuous line numbers.

To regenerate it and test compilation after extraction into a clean directory:

```sh
python release/build_arxiv.py --check
```

`results/arxiv_bundle.json` records the archive and member hashes and the
two-pass clean compilation. Install the documented TeX packages before this
check. The local build test does not submit anything to arXiv.

For this working revision the output is
`dist/neurosymbolic-arxiv-claude-revision.tar.gz`. The dependency scanner
collects the current manuscript, both generated numerical inputs, generated
tables, and only the figures actually included. It checks a reviewed flat-file
allowlist, normalized archive metadata, input hashes, and clean PDFLaTeX passes until references stabilize
(at least two and at most four), with shell escape disabled. The earlier published bundle is unchanged.

## External-review revision studies

`revision_experiments.py` runs the matched raw/normalized and fixed/coupled
operating-grid ablation, constrained variants, a constructive scale-chase
illustration, short-horizon sensitivity diagnostics, and a pendulum control.
Their configurations and every seed outcome are saved in
`results/revision_experiments.json`, with timings in `results/timing_revision.json`.

`revision_extensions.py` supplies general-dimensional exact polynomial algebra
and an outward adaptive matrix oracle, a coupled three-dimensional synthesis
benchmark, matched-candidate oracle checks, and a polynomial surrogate fitted
to trajectories. The known polynomial truth is used to certify the residual
and Jacobian globally; sampled fitting errors alone are not residual certificates.
Adaptive proof budgets can return inconclusive and do not remove worst-case
dimensional complexity. Results are in `results/revision_extensions.json`.

`revision_report.py` builds the full per-certified-seed constant table and timing
summaries in `results/revision_report.json`. `fill_tex.py` checks its input hashes
and writes `paper/revision_numbers.tex` and `paper/revision_tables.tex` alongside
the original numerical macros. Componentwise medians describe distributions;
only complete per-candidate rows supply joint guarantees.

## Scientific outputs and provenance

| Producer | Output | Role |
| --- | --- | --- |
| `experiments.py` | `results/traces.json` | Three regimes over the same 20 seeds |
| `experiments.py` | `results/gridcost.json` | Sampled 2-D slack ratios and dimension-cost extrapolation |
| `residual.py` | `results/residual.json` | Sufficient residual budget and measured perturbation sweep |
| `bench_pendulum.py --experiment` | `results/pendulum.json` | Non-polynomial dynamics benchmark |
| `bench_neural.py --experiment` | `results/neural.json` | Six-unit neural metric benchmark |
| `smoke_p1.py` | `results/smoke_p1.json` | Reverification at the producer's grid and subdivision depth |
| `fill_tex.py` | `paper/numbers.tex` | Validated numbers and source-content hashes |

Each experiment regenerates its associated figures through Matplotlib.
`fill_tex.py` requires the complete set of production results and fails if a
required input is absent or invalid. Edit numerical code or configuration and
rerun the producers, then the fill script; do not hand-edit measured values in
the manuscript or `numbers.tex`.

Random seeds fix the sampling and initialization. Scientific values can still
depend on NumPy, BLAS, hardware, and arithmetic order through optimizer paths.
Timings and run timestamps vary on every run; PDF metadata can also change.
Compare scientific fields rather than expecting byte-identical JSON or PDFs
across environments. The run manifest and separate timing records document
these differences.

Scientific result files, figure PDFs, the paper PDF, and provenance records
belong in the research snapshot. `.gitignore` excludes caches and transient
LaTeX files, and deliberately permits `results/` and its logs. This workflow
does not create a Git commit automatically.

## Scope and interpretation

All implemented benchmarks have two state variables. Higher-dimensional grid
costs extrapolate a measured 2-D ratio while holding it fixed; they are not
experiments on higher-dimensional dynamics. Differential guarantees apply to
solutions of the variational equation along trajectories remaining in the
certified domain. A finite two-trajectory bound additionally needs an initial
connecting curve whose entire flow remains in that domain. Convex forward
invariance is a sufficient condition for the displayed Euclidean overshoot
bound. Two trajectories remaining in the box alone do not imply that condition.
The box in the polynomial benchmark is not forward invariant, and no global
incremental-convergence claim is made for it.

The scale-chase result is a theoretical degeneracy under a scale-closed family,
global strict negativity of the fixed shape, a positive scale-equivariant
Lipschitz bound, and initial noncertification. Negativity on current data alone
does not support an infinite scale chase after data augmentation. The normalized
coarse-grid experiment exhibits growth of the shape-dependent certified
Lipschitz bound for the normalized test as the coupled target outgrows the
achieved slack. The corrected loop has no general
termination or relative-completeness guarantee. Control contraction metrics
remain outside this paper's implemented scope.

The residual theorem supplies a sufficient budget. A negative sampled
contraction condition beyond that budget does not certify the whole domain or
locate a failure threshold. An unsuccessful metric search does not establish
that the hypothesis class is infeasible.

The verifier treats stored binary64 parameters as exact real inputs. Polynomial
coefficients, derivatives, and recentering use exact rational arithmetic before
conversion to outward endpoints. Interval operations widen each elementary
binary64 operation in the required direction. Transcendental enclosures use
MPFR through gmpy2, with rigorous interpolation remainders for large batches.
They do not assume an error bound for NumPy's transcendental functions.

Ordinary floating-point evaluations are a rejection screen. Returned training
points are diagnostic candidates; rejection is not a proof of a contraction
violation. A positive result
requires outward upper eigenvalue/trace-quotient enclosures at every grid node,
an outward bound on the covering radius from the largest actual node gap, and
an upward-rounded Lipschitz margin. Certified slack and sufficient residual
budgets are rounded downward. Every positive result records that the grid was
enclosed. The neural verifier can prove groups of grid nodes together using
adaptive interval boxes and subdivides unresolved groups down to individual
nodes; it retains the original grid and Lipschitz margin. Nonfinite arithmetic
and invalid interval operations fail closed.

The trusted arithmetic assumptions are IEEE binary64 round-to-nearest with
gradual underflow and MPFR's directed-rounding contract. The arithmetic
acceptance tests check these platform properties and compare enclosures with
exact rational and higher-precision calculations. Dense sampling and derivative
tests remain diagnostics rather than proofs of enclosure soundness.

## Code and historical notes

`cegis_c.py` supplies the polynomial dynamics, metric, learner, and default
bound-based oracle. `bounds.py` builds exact polynomial coefficient bounds;
`rounding.py` supplies outward elementary operations, MPFR transcendental
enclosures, and arithmetic platform checks.
`experiments.py`, `residual.py`, and the two `bench_*.py` scripts are the current
paper producers. `run_all.py` is the authoritative reproduction entry point.
The experiment JSON files record operational outcome definitions, per-study
round caps, and training settings. Neural certificates also export outward
metric bounds and conservative overshoot and improved differential-rate bounds.
`package_reproduction.py` packages and checks the completed run for reviewers.

`figures.py` and `cert_ladder.py` are retained historical prototype/diagnostic
scripts. The current figure producers are listed above.
`RESPONSE_TO_REFEREE.md` records the manuscript's revision response.
`release/build_editorial.py` validates reuse of the verified scientific
snapshot and rebuilds the current paper; `release/build_arxiv.py` packages
the manuscript inputs and checks their standalone compilation.

Run focused regression checks without rerunning the experiments:

```sh
python -m unittest discover -s tests -v
```
