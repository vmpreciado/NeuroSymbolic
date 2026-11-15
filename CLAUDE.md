# CLAUDE.md — CEGIS-C for contraction certificates

## What this repository is

Code and manuscript for the paper "Neurosymbolic Synthesis of Contraction
Metrics with Certified Verification" (V. M. Preciado). The code implements CEGIS-C: a
counterexample-guided loop that trains a state-dependent contraction metric
and certifies it with a sound grid oracle. The manuscript is
`paper/nsp_contraction_paper_ieee.tex` (IEEEtran preprint, single column,
without reviewer line numbers). README.md
describes the current reproduction workflow; PLAN.md, REVISION_PLAN.md, and
P4_FINDINGS.md are historical records rather than a pending task list.

## Repository layout

```
cegis_c.py        core dynamics, metric class, learner, and grid oracle
bounds.py         polynomial bounds and acceptance checks
rounding.py       outward interval operations, MPFR enclosures, platform checks
experiments.py    current multi-seed regimes and figures 1-3
residual.py       Theorem 2 experiment and figure 4
bench_pendulum.py non-polynomial benchmark and figure 5
bench_neural.py   neural metric benchmark and figure 6
smoke_p1.py       independently reverify a certified operating-grid candidate
fill_tex.py       validate result inputs and generate paper/numbers.tex
run_all.py        full reproduction, logs, and source/environment provenance
package_reproduction.py hash-verified local reviewer archive after a passing run
results/         scientific JSON outputs, timings, and run records
paper/           manuscript, generated numbers, figures, and compiled PDF
figures.py       historical prototype, not the current figure pipeline
cert_ladder.py   historical diagnostic, not part of run_all.py
```

## Mathematical context (do not rederive; trust these)

State dimension n=2, domain X = [-2,2]^2, dynamics
f(x) = (-x1, -(x2 - x1^2)), target rate RATE = 0.4.

Metric class: M_theta(x) = ALPHA*I + L(x)^T L(x), ALPHA = 0.1,
L lower triangular, entries linear in features (1, x1, x2),
theta in R^{3x3} (rows = L11, L21, L22 coefficient vectors).

Contraction matrix:
C(x) = Mdot(x) + A(x)^T M(x) + M(x) A(x) + 2*RATE*M(x),
A = Jacobian of f, Mdot = sum_k (dM/dx_k) f_k.

Polynomial degrees (n=2, these features, this f): entries of M are
degree <= 2; entries of A are degree <= 1; entries of C are degree <= 3;
entries of dC/dx_k are degree <= 2; tr M is degree <= 2;
entries of dM/dx_k are degree <= 1. Everything in this pipeline is a
polynomial with coefficients that are explicit in theta.

Test functions: g(x) = lambda_max(C(x)) (raw),
gt(x) = g(x)/tr M(x) (normalized; the oracle and learner use this one).

Grid oracle (sound iff L is a TRUE Lipschitz upper bound): certify iff
gt(x_i) <= -L*tau on all grid points, tau = covering radius.
The default verifier uses bounds.py and rounding.py. `lipschitz_estimate`
remains an EMPIRICAL diagnostic and never supports a certificate. Positive
answers require outward-rounded bounds, actual-grid coverage, and interval
grid values. Polynomial construction uses exact rational coefficients;
transcendental endpoints use MPFR through gmpy2. Nominal rejection screens
cannot certify. State IEEE binary64/gradual-underflow and MPFR assumptions;
do not describe sampling tests as a proof of enclosure.

## Hard rules

1. NEVER fabricate a number. Every number in the manuscript must be read
   from a JSON file in results/ that a script in this repo wrote.
   The fill script (Task 2) is the only thing that edits numbers in the tex.
2. NEVER weaken soundness silently. If certification fails with the
   certified Lipschitz bound, do not lower the bound, shrink the domain,
   or raise ALPHA to make it pass. Follow the contingency in PLAN.md
   (finer grid first, then subdivision depth), and if it still fails,
   STOP and write results/FAILURE_REPORT.md describing exactly what failed
   and the measured gap.
3. All randomness seeded (npr.seed). A rerun in the SAME numerical environment
   must reproduce the scientific values up to float formatting; timings and
   run metadata vary. Cross-platform
   bitwise reproducibility is NOT expected (BLAS and FP differences steer
   the optimizer); this is exactly why every claim about loop behavior is
   statistical over seeds (PLAN.md Task 2). Never present a single-seed
   outcome as the behavior of a variant.
4. Use autograd + numpy + matplotlib for the learner and experiments, and
   gmpy2/MPFR for directed verification arithmetic (explicitly requested in
   the outward-rounding revision). No torch, no jax, no sympy
   unless Task 1's hand-rolled polynomial class fails its tests, in which
   case sympy is the approved fallback for constructing coefficients
   (bounds must still be computed by bounds.py, not by sympy heuristics).
5. Manuscript style (strict): active voice, short direct sentences,
   no em dashes anywhere, never the words "notably" or "importantly",
   no bullet lists in prose, commas and colons to structure ideas.
6. Preserve theorem statements and proofs unless the user authorizes a
   mathematical revision. The user authorized implementation of the external
   review plan, including corrected hypotheses, an idealized finite-grid
   termination proposition, general-dimensional bounds, and new experiments.
   Investigate numerical contradictions rather than changing a theorem to fit
   experimental values. Differential contraction does not imply finite pairwise
   convergence without the connecting-flow/domain hypotheses in the manuscript.
7. LaTeX must compile with zero errors and zero undefined references:
   run pdflatex twice and grep the log before declaring any tex task done.
8. Figures: regenerate via scripts only, never hand-edit PDFs. Fonts and
   sizes must match the existing figures (matplotlib defaults as in
   figures.py).

## Definition of done (current reproduction)

- Bound and derivative acceptance checks pass before benchmark outputs count.
- All production scientific results and included figures regenerate from the
  documented seeded configuration; historical diagnostics stay labeled.
- Smoke verification certifies at the same grid and depth as its producer.
- fill_tex.py validates all required inputs and records their content hashes.
- Two successful LaTeX passes produce the current PDF without errors or
  undefined references/citations; inspect the rendered pages.
- `python3 run_all.py` succeeds end to end, with current logs and a manifest
  recording source/artifact hashes, environment, and stage outcomes.
- All fifteen declared stages pass, including the new studies, the analytical
  constant-metric baselines, complete
  reporting, and a clean arXiv bundle compilation. The fixed-budget Adam
  implementation is not covered by the idealized termination proposition.
- `python3 package_reproduction.py` produces the complete checked local reviewer
  archive after a fresh full run. The v1.2.2 revision uses
  `release/build_application_packet.py` after a fresh full passing run. The
  older v1.1.1 editorial-overlay scripts are historical and cannot package
  newly generated scientific evidence.

## Current application and constant-metric revision

`application_protocol.json` fixes the swing-model comparison. The new module
`application_verifier.py` verifies the shared neural metric;
`application_baseline.py` independently implements a documented adaptation of
Fitzsimmons–Liu Eq. (5), not the original full SMT workflow.
`application_comparison.py` retains every prespecified outcome and proof;
`application_report.py` renders validated results through `fill_tex.py`.
The invariant energy region is the sublevel restricted to the verification
box. Do not claim the unrestricted global tilted-potential sublevel is compact.

`application_constant_baseline.py` verifies a fixed analytical constant metric
on all three application boxes using directed transcendental bounds and exact
rational matrix checks. This candidate was added after the neural comparison;
it has no training seeds and does not constitute an optimized constant-metric
search or a runtime comparison. Its success on the smallest box means neural
state dependence is unnecessary for that result. Its pointwise violations on
the larger box boundaries rule out this matrix for the whole-box condition;
they do not rule out other constant metrics or contraction on the invariant
interior energy sublevels. Preserve every original neural outcome and budget.

`polynomial_constant_baseline.py` independently proves the fixed post hoc
matrix diag(12,1) on the polynomial benchmark at the requested rate. Its exact
endpoint proof covers decimal and binary64 rate conventions and does not run
the learned-metric grid oracle. It has no training seeds, optimized search, or
matched timing comparison. State dependence is unnecessary for this benchmark
at this rate. Preserve the original learning counts, budgets, and the
non-invariance/connecting-flow qualification.
