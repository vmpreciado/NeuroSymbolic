# Neurosymbolic Synthesis of Contraction Metrics with Certified Verification

This is the **v1.2.2** reproduction and manuscript, available from the
[immutable release](https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.2.2).
It harmonizes descriptive rounding and explains the polynomial constant-metric
choice and its independently selected proof margin. It retains the published two-machine
power-system model, matched Fitzsimmons–Liu PDI-loss adaptation, fixed-data hinge
control, and verified swing-model constant metric. The paper has no reviewer line
numbers. The previous immutable
[v1.2.1 release](https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.2.1)
remains unchanged. See the release
record for publication status and asset checksums. This workflow does not
submit the paper to arXiv.

## Reproduce all scientific results and the paper

Use Python 3.12.7 and the locked dependencies. A TeX installation must provide
PDFLaTeX, IEEEtran, amsmath, amssymb, amsthm, algorithm, algpseudocode, graphicx,
hyperref, xcolor, and placeins.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python run_all.py --parallel-science
```

All fifteen stages must pass. The runner executes arithmetic/regression tests,
the original polynomial, residual, pendulum and neural studies, smoke checks,
the controlled revision experiments, adaptive three-dimensional and surrogate
extensions, the matched power-system application, the analytical constant-metric
baselines for both the swing and polynomial models, complete reporting,
generated manuscript numbers/tables, LaTeX compilation, and a fresh extracted
arXiv-source compilation. Independent legacy producers can overlap; the new
application runs after they have completed, with serial method timings.

The runner records exact input/output hashes, environment, complete logs and
stage outcomes in `results/run_manifest.json`, `results/run_all.log`, and
`results/stage_logs/`. It stops on a failed stage or invalid/stale output.
Training paths can differ across numerical environments; the seed, environment,
parameters, and positive proof records are retained. Positive certificates
require directed arithmetic and complete verification; samples cannot certify.

## New application and comparison

`application_protocol.json` fixes every domain, seed, target, learning and
verification budget. `application_comparison.py` runs all three learners.
`application_baseline.py` independently implements the spectral PDI objective
from Fitzsimmons and Liu (2024), Eq. (5). The source mapping, original protocol,
and explicit adaptations are in `comparison_sources/fitzsimmons_liu_2024.json`.
No original unlicensed source code is included.

All arms share the neural metric class, initialization, initial collocations,
raw training target, requested contraction rate, optimizer-update caps, and
outward verification budget. A fixed-data hinge control isolates diagnostic
feedback from the loss choice. The comparison is an adaptation of a published
learning objective, not a reproduction or ranking of the original LyZNet/dReal
toolchain. No per-method tuning or post-outcome domain shrinking is performed.
Every prespecified outcome is retained, including unsuccessful runs.
An explicitly exploratory secondary analysis was selected after preliminary
primary outcomes: all frozen final candidates are reverified with a common
65,535-box budget and no change to parameters, domains, margins or learning.
This diagnostic reports added verification cost and leaves primary outcomes
unchanged; it is not a prespecified confirmatory performance comparison.

`application_verifier.py` uses exact-real physical constants, directed MPFR
transcendental bounds, adaptive complete binary partitions, and outward matrix
checks. It saves and replays every positive leaf. Energy sublevels restricted
to the verification box give forward-invariant regions for interpreting
successful certificates as convergence to the synchronized equilibrium.
Exhausted budgets and nominal rejection remain inconclusive. The model is
physically motivated; it is not operational grid data or hardware validation.

```sh
python application_comparison.py
python application_comparison.py --validate
```

### Analytical constant-metric baseline

`application_constant_baseline.py` checks the fixed matrix
`P = [[1, 1/2], [1/2, 2]]` on every prespecified application box. Directed MPFR
trigonometric enclosures and exact rational principal-minor checks establish
the small-box contraction inequality at the same requested rate and raw margin
as the neural comparison. The complete deterministic proof and its replay are
stored in `results/application_constant_baseline.json`.

This analytical candidate was introduced after inspecting the neural study.
It uses no training or seeds and is not an optimized search over all constant
metrics or a matched runtime experiment. Its small-box success shows that this
application result does not require a neural or state-dependent metric.
On each larger box, a certified pointwise violation rules out this fixed matrix
for the whole-box condition. It does not rule out other constant metrics or
contraction on the interior invariant energy subset.

```sh
python application_constant_baseline.py
python application_constant_baseline.py --validate
```

### Polynomial constant-metric baseline

`polynomial_constant_baseline.py` proves that `P = diag(12, 1)` certifies the
original polynomial benchmark on `[-2,2]^2` at rate `0.4`, with exact raw margin
`1/20` and normalized margin `1/260`. Exact endpoint matrix checks cover both
the decimal rate `2/5` and the actual stored binary64 rate. The proof is saved
in `results/polynomial_constant_baseline.json` and replayed by the pipeline.
For `P = diag(p,1)`, strict contraction requires `p > B^2/(1-lambda)^2`;
the threshold is `100/9` here, so `12` is a simple integer choice. The raw
proof margin is selected independently of the normalized learner margin;
separate shifted endpoint checks establish its value.

This is another fixed post hoc analytical certificate, without training seeds,
optimized search, or a matched runtime comparison. The endpoint proof differs
from the learned-metric grid oracle. It shows that state dependence is
unnecessary for this benchmark at the requested rate; the original experiments
remain diagnostics of synthesis and verification, with unchanged budgets.
The benchmark box is not forward invariant; finite-distance conclusions retain
the paper's connecting-flow conditions.

```sh
python polynomial_constant_baseline.py
python polynomial_constant_baseline.py --validate
```

The standalone development pilot uses excluded seed 1000 and writes only under
`tmp/`. It cannot enter the paper or pass the production validator.
`fill_tex.py` validates the scientific results and calls `application_report.py`
to generate the application numbers and tables. Never hand-edit generated data.

## Rebuilt artifacts and independent review

After the complete pipeline passes, with no source or result changes, install
Poppler and put `pdftotext` on PATH, then run:

```sh
python release/build_application_packet.py
```

The outputs are:

- `dist/neurosymbolic-paper-v1.2.2.pdf`
- `dist/neurosymbolic-arxiv-v1.2.2.tar.gz`
- `dist/neurosymbolic-reproduction-v1.2.2.zip`
- `dist/neurosymbolic-claude-deep-review-v1.2.2.zip`
- `dist/arxiv_abstract.txt`
- `dist/neurosymbolic-claude-review-prompt.txt`

The arXiv archive contains only the main TeX, its generated TeX dependencies,
and referenced PDF figures, with an embedded bibliography. It is compiled in
an empty directory with shell escape disabled and must have no warnings or
unresolved references. The reproduction ZIP contains the complete manifest-
verified source, scientific artifacts and stage logs of this fresh run.

The Claude ZIP includes the current PDF, text export, source archive, complete
runnable project, full reproduction ZIP, and a neutral independent-review
prompt. Extract it and run its standard-library `VERIFY_PACKAGE.py` to check
all hashes, the passing-run evidence, and nested reproduction integrity without
executing scientific code. Checksums establish file integrity, not mathematical
correctness. Prior referee material in `project/reviews/` is optional history.

Packaging rejects stale outputs, changed sources, incomplete runs, or edited
proof data. It does not publish anything or overwrite immutable release assets.
The older `release/build_current.py` and `release/package_current.py` describe
the v1.1.1 editorial overlay and must not be used to package this new science.

## Scope and limitations

The original matched polynomial study did not demonstrate a certification-count
advantage from normalization or margin coupling. Practical Adam has no proved
termination guarantee. The constructed scale chase requires a scale-closed
class, which the main fixed-floor class generally lacks. Three-dimensional and
neural results remain finite-budget demonstrations. Surrogate transfer uses
known truth to certify global residual and Jacobian bounds; it is not a
finite-data guarantee for unknown noisy dynamics. The new physical comparison
has its own explicitly limited common-class/common-verifier scope. The paper
acknowledges AI assistance and retains author responsibility for all claims.
