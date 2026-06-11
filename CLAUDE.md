# CLAUDE.md — NeuroSymbolic (CEGIS-C for contraction certificates)

## What this repository is

Code and manuscript for the paper "Neurosymbolic Synthesis of Contraction
Certificates" (V. M. Preciado). The code implements CEGIS-C: a
counterexample-guided loop that trains a state-dependent contraction metric
and certifies it with a sound grid oracle. The manuscript is
`paper/nsp_contraction_paper_ieee.tex` (IEEEtran, single column, lineno,
draft watermark).

## Repository layout

```
cegis_c.py        core: dynamics, metric class, test functions, learner,
                  grid oracle, CEGIS loop
figures.py        experiment runner + figure generation (figs 1-3)
bounds.py         (TO BE CREATED, Task 1) certified polynomial bounds
residual.py       (TO BE CREATED, Task 3) Theorem 2 experiment (fig 4)
results/          JSON files of measured numbers, written by scripts
paper/            nsp_contraction_paper_ieee.tex + fig_*.pdf
PLAN.md           the task list; execute tasks in order
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
The current `lipschitz_estimate` in cegis_c.py is an EMPIRICAL sampled
estimate with a 1.5 safety factor. Replacing it with a certified bound is
Task 1 and is the point of this work session.

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
3. All randomness seeded (npr.seed). A rerun of any script on the SAME
   machine must reproduce the JSON up to float formatting. Cross-platform
   bitwise reproducibility is NOT expected (BLAS and FP differences steer
   the optimizer); this is exactly why every claim about loop behavior is
   statistical over seeds (PLAN.md Task 2). Never present a single-seed
   outcome as the behavior of a variant.
4. Keep autograd + numpy + matplotlib only. No torch, no jax, no sympy
   unless Task 1's hand-rolled polynomial class fails its tests, in which
   case sympy is the approved fallback for constructing coefficients
   (bounds must still be computed by bounds.py, not by sympy heuristics).
5. Manuscript style (strict): active voice, short direct sentences,
   no em dashes anywhere, never the words "notably" or "importantly",
   no bullet lists in prose, commas and colons to structure ideas.
6. Do not touch the theorem statements or proofs in the tex. If a measured
   quantity contradicts a theorem, that is a bug in the code or a soundness
   violation; investigate, do not edit the theorem.
7. LaTeX must compile with zero errors and zero undefined references:
   run pdflatex twice and grep the log before declaring any tex task done.
8. Figures: regenerate via scripts only, never hand-edit PDFs. Fonts and
   sizes must match the existing figures (matplotlib defaults as in
   figures.py).

## Definition of done (whole session)

- bounds.py exists, passes its acceptance tests (PLAN.md Task 1).
- cegis_c.py verifier uses the certified bound; safety-factor language
  removed from code comments.
- Figures 1-3 regenerated with certified constants; results/*.json updated.
- residual.py produces fig_residual.pdf and results/residual.json.
- Manuscript: footnote replaced (exact text in PLAN.md Task 4), new
  subsection V-D inserted (skeleton in PLAN.md Task 3), all [XX]
  placeholders filled by the fill script, zero \todo markers remain,
  compiles clean.
- `python3 run_all.py` (create it: runs everything in order) succeeds
  end to end from a clean checkout.
