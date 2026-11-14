# Neurosymbolic Synthesis of Contraction Metrics with Certified Verification

This release addresses the November 2026 external referee report and the final novelty/comparison review.
The neural extension couples a learned metric factor to explicit contraction
conditions, analytic/exact algebra where applicable, and outward-rounded
verification. The main controlled ablations use fixed polynomial features;
the neural parametrization is evaluated separately.
The introduction distinguishes certificate search for a fixed vector field
from contracting-model architectures and partitioned neural controller/metric
training, with explicit comparisons to Jaffe et al. and Davydov.
Release [v1.1.0](https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.1.0)
archives the current manuscript and verified reproduction. It was developed on
`codex/claude-revision` from immutable release v1.0.1. `RESPONSE_TO_CLAUDE.md` records every review disposition;
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
`amssymb`, `amsthm`, `algorithm`, `algpseudocode`, `graphicx`,
`hyperref`, `xcolor`, and `placeins`. PDF inspection additionally uses Poppler's `pdfinfo`
and `pdftoppm`; these are not needed by the experiment scripts. Building the
Claude review package additionally requires Poppler's `pdftotext` on `PATH`.

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
The current paper uses a single-column, single-spaced preprint layout without
reviewer line numbers. Feedback can cite pages and section or theorem labels. It retains the
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

[Immutable release v1.1.0](https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.1.0)
contains the current paper without reviewer line numbers, the tested arXiv source
bundle, the complete current reproduction, and the Claude independent-review
package. `CITATION.cff` and reference [18] in the paper identify this release.
The earlier immutable v1.0.0 and v1.0.1 releases remain unchanged.

The scientific evidence comes from a passing twelve-stage run. Its exact original
archive is preserved byte-for-byte as `neurosymbolic-reproduction-scientific-verified.zip`
with SHA256 `c1cdea042cca4f5495a3ed025174570bb68e494df2695641af2e97d91437eb39`.
Its manifest, stage logs, learned parameters, proof records, figures, and numerical
results are unchanged. Its embedded manuscript predates the final editorial edits.

`neurosymbolic-reproduction-v1.1.0.zip` is the complete current runnable project.
It overlays the final manuscript and metadata on the unchanged scientific files,
preserves the original scientific archive under `provenance/`, and provides
`SNAPSHOT.json` and a standalone `VERIFY_CURRENT_REPRODUCTION.py`. After extraction,
run that verifier with Python 3.12 before rebuilding. The verifier checks hashes,
original run evidence, and the declared editorial changes; it does not establish
mathematical correctness or rerun experiments.

`results/editorial_v1.1.0.json` records the source changes, reused scientific
snapshot, saved-result validation, new manuscript compilation, and clean arXiv
build. This release does not claim a new full experimental run. To create a new
scientific run, use `run_all.py`; the original strict `package_reproduction.py`
requires a fresh passing run with all current inputs. To reconstruct the released
editorial overlay from saved results, run:

```sh
python release/build_current.py
python release/package_current.py
python release/build_review_packet.py
```

These commands rebuild the editorial outputs from the verified scientific inputs.
They do not publish a GitHub release. `release/SHA256SUMS` lists the release assets.

## arXiv source bundle

Release asset `neurosymbolic-arxiv-v1.1.0.tar.gz` contains thirteen flat manuscript
inputs: the main TeX source with embedded bibliography, both generated numerical
inputs, generated tables, and nine included figure PDFs. It contains no compiled
manuscript, logs, experiment data, or nested archive. The manuscript uses a
single-column, single-spaced layout without reviewer line numbers.

To regenerate it and test compilation after extraction into a clean directory:

```sh
python release/build_arxiv.py --check --output dist/neurosymbolic-arxiv-v1.1.0.tar.gz
```

`results/arxiv_bundle.json` records the member hashes and clean compilation.
The dependency scanner checks a reviewed flat-file allowlist, normalized archive
metadata, and PDFLaTeX passes until references stabilize, with shell escape disabled.
This local test does not submit anything to arXiv. The earlier published source
bundles remain unchanged.

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

The original polynomial, pendulum, and neural studies have two state variables.
The revision adds a coupled three-dimensional benchmark with adaptive verification.
Uniform-grid cost extrapolations separately hold a measured 2-D ratio fixed; those
extrapolated counts are not higher-dimensional experiments. Differential guarantees apply to
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
`release/build_editorial.py` preserves the historical v1.0.1 editorial workflow.
`release/build_current.py` validates reuse of the twelve-stage snapshot and rebuilds
the current manuscript; `release/build_arxiv.py` checks standalone compilation.
`release/package_current.py` creates the verified current reproduction overlay,
and `release/build_review_packet.py` creates the independent-review package.

Run focused regression checks without rerunning the experiments:

```sh
python -m unittest discover -s tests -v
```
