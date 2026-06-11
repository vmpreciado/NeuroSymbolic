# PLAN.md — work session: certified Lipschitz bound + residual experiment

Execute tasks in order. Each task has acceptance criteria; do not start the
next task until the current one passes. Read CLAUDE.md first.

---

## Task 1: certified Lipschitz bound (bounds.py)

Goal: replace the sampled Lipschitz estimate in `cegis_c.py` with a sound
upper bound, so Proposition 1 of the paper holds unconditionally for the
reported experiments.

### 1a. Polynomial module

Implement a minimal polynomial class over n=2 variables:
representation `dict[(i, j)] -> float` meaning coeff * x1^i * x2^j.
Operations: add, sub, scalar mul, poly mul, partial derivative,
evaluation at a point (for tests), and `supabs_box(B)`:

    supabs_box(p, B) = sum_{(i,j)} |c_{ij}| * B^(i+j)

which upper-bounds sup_{x in [-B,B]^2} |p(x)|. This is the coefficient
bound; it is sound and possibly loose.

Refinement (required): `supabs_subdiv(p, B, depth)`: split [-B,B]^2 into
4^depth equal subboxes; on each subbox with center c and half-width h,
recenter the polynomial (substitute x -> c + y, expand; degree <= 3 so this
is a small exact computation) and apply the coefficient bound with B := h;
return the max over subboxes. Soundness is inherited; tightness improves
monotonically with depth. Default depth 3, configurable.

### 1b. Symbolic assembly of C and its derivatives

Build, as polynomial objects with numeric theta plugged in:
entries of L, M (= ALPHA*I + L^T L), dM/dx_k, Mdot (= sum_k dM/dx_k * f_k
with f1 = -x1, f2 = -(x2 - x1^2) as polynomials), A entries
(A11=-1, A12=0, A21=2*x1, A22=-1), and finally the three independent
entries of the symmetric C: c11, c12, c22. Then dC/dx_k entrywise.

Acceptance test 1b: for 200 random (theta, x) pairs, the polynomial
evaluation of c11, c12, c22 matches the numeric values computed by the
existing vectorized code in cegis_c.py to 1e-10 relative. Same for tr M.
This test is mandatory; it is the firewall against transcription bugs.

### 1c. The certified constants

For the raw test function g = lambda_max(C):

    Lg <= sqrt( sum_k Dk^2 ),
    Dk  = sqrt( supabs(dc11/dx_k)^2 + 2*supabs(dc12/dx_k)^2
              + supabs(dc22/dx_k)^2 )            # Frobenius bound on dC/dx_k

(uses: lambda_max is 1-Lipschitz in spectral norm; ||.||_2 <= ||.||_F;
Cauchy-Schwarz over coordinates).

For the normalized test gt = g / tr M, with v = tr M:

    v_min  = n*ALPHA = 0.2                       # exact, by construction
    u_max <= sqrt( supabs(c11)^2 + 2*supabs(c12)^2 + supabs(c22)^2 )
    Lv    <= sqrt( supabs(dv/dx1)^2 + supabs(dv/dx2)^2 )
    Lgt   <= Lg / v_min + u_max * Lv / v_min^2

All supabs via supabs_subdiv at the configured depth.

Acceptance test 1c: for 10 random theta and 20 fine-grid samplings each,
the certified Lgt is >= the empirical finite-difference estimate (without
any safety factor). Also: certified Lgt at depth 4 <= certified Lgt at
depth 2 (monotone improvement), for 5 random theta.

### 1d. Wire into the verifier

Add `lipschitz_certified(theta, depth=3, normalized=True)` and make
`verify()` use it by default; keep the empirical estimator available under
a flag for comparison plots only. Remove the LIP_SAFETY language from the
docstrings of the certified path.

Contingency: the certified bound will be looser than the empirical
estimate, so certification of the final candidate may fail at the 121^2
grid. Escalate in this order, logging each step to results/cert.json:
(i) verification grid 241^2, then 481^2 (cost is trivial in 2D);
(ii) subdivision depth 4, then 5;
(iii) if still uncertified, run the full CEGIS loop with the certified
oracle in the loop (it may simply need one or two more rounds since the
required margin grew). Only if all three fail, write FAILURE_REPORT.md
and stop (CLAUDE.md rule 2).

---

## Task 2: multi-seed experiments with certified constants

Background (read first): single-seed runs of the three variants are NOT
platform-stable. On x86/OpenBLAS the unnormalized coupled variant diverged
(L: 3.3 -> 9.2e3 over 12 iterations); on Apple Silicon/Accelerate it
certified in 3 iterations (L: 3.3 -> 4.0). Fixed-margin stalled and
normalized-coupled certified on both. The scale-invariant ratio slack/L
agreed across platforms to <1% (0.0474 vs 0.047), as Lemma 1 predicts.
Consequence: all claims about variant behavior must be statistics over
seeds, computed on THIS machine, and the manuscript must not assert that
any single trajectory is "the" behavior.

### 2a. Protocol

Run each of the three variants (fixed-margin; coupled unnormalized;
coupled normalized) with the certified oracle for seeds 0..19. The seed
controls the initial dataset, perturbation noise, and any learner
randomness. Per seed and variant record: certified (bool), iterations to
certify (or cap), max L over iterations, initial L, final certifiability
gap. Define DIVERGED := (not certified) and (max L > 100 * initial L).
Define STALLED := (not certified) and (final gap within 10% of the gap at
iteration 2). Write results/traces.json with the full per-seed table plus
summary: per-variant counts certified/diverged/stalled, median and IQR of
iterations-to-certify for the normalized variant, max observed L growth
factor for the unnormalized variant.

### 2b. Figure 2 (replace)

Same axes as before (certifiability gap vs iteration, log y). Plot every
seed as a thin line (alpha ~0.3), median per variant as a bold line, three
colors as before. Legend reports the counts, e.g. "normalized coupled
(20/20 certified)". Must remain legible at 0.72 textwidth; if 60 thin
lines are unreadable, drop to per-variant 10-90 percentile bands plus
median, and say so in the caption.

### 2c. Grid-cost sweep

As before, but compute the slack/L ratio with certified constants for
every certified normalized-variant seed and use the MEDIAN ratio for the
sweep; report min/max ratio across seeds in results/gridcost.json.

### 2d. Manuscript edits (placeholders first, then fill_tex.py)

Replace the affected sentences with the following, placeholders filled
from results/*.json (adjust grammar only if a count forces it; keep the
style rules of CLAUDE.md):

(i) In the paragraph after Corollary 2, replace the clause saying the
scale chase "is the one taken in practice, with the measured Lipschitz
constant growing by three orders of magnitude across twelve iterations"
with: "and Section V shows it is taken in [XX:naive_diverged] of
[XX:n_seeds] runs on our benchmark, with Lipschitz growth factors up to
[XX:naive_maxgrowth]; in the remaining runs the learner reaches a
certifiable shape before the chase dominates, so the failure is
probabilistic, which is itself an argument against the unnormalized loop."

(ii) In Section V, replace the single-trajectory description of the three
variants with the seed statistics: fixed-margin stalls in
[XX:fixed_stalled] of [XX:n_seeds] runs and certifies in none;
unnormalized coupling certifies in [XX:naive_cert] and diverges in
[XX:naive_diverged] of [XX:n_seeds]; normalized coupling certifies in
[XX:norm_cert] of [XX:n_seeds] with median [XX:norm_med_iters]
retraining rounds.

(iii) Abstract: replace "certification by the normalized margin-coupled
loop in one retraining round" with "certification by the normalized
margin-coupled loop in all seeds within a median of
[XX:norm_med_iters] retraining round(s)", and replace "the stall, the
scale chase" phrasing only if the counts make it inaccurate.

(iv) Add one sentence at the end of the Figure 3 paragraph: "The
slack-to-Lipschitz ratio is the stable quantity across seeds, with range
[[XX:ratio_min], [XX:ratio_max]], consistent with Lemma~\ref{lem:scale}."

Acceptance: figures 1-3 regenerated from this machine's runs; every
number in the tex traceable to results/*.json; paper compiles clean.

---

## Task 3: residual experiment (Theorem 2) and Figure 4

Goal: exercise Theorem 2 computationally. The theorem: if the metric is
certified for the model f with uniform margin eta, and the true dynamics
are f + d with ||d|| <= delta0, ||dd/dx||_2 <= delta1 on X, and
eta > mu1*delta0 + 2*b*delta1, then the true system contracts at rate
RATE + (eta - mu1*delta0 - 2*b*delta1)/(2b).

### 3a. Certified ingredients (extend bounds.py)

- eta: uniform margin of the final certified candidate on the RAW test,
  eta = min_i(-g(x_i)) - Lg*tau, computed on the finest grid used in
  Task 1d's escalation. If eta <= 0.02, refine the grid until eta
  stabilizes; report eta in results/cert.json.
- b_upper: certified upper bound on lambda_max(M) over X via
  sqrt(supabs(m11)^2 + 2*supabs(m12)^2 + supabs(m22)^2)... use the tighter
  b_upper = supabs-based Frobenius bound on M; record it.
  (a = ALPHA = 0.1 exactly.)
- mu1: certified bound on sup_x sqrt(sum_k ||dM/dx_k||_2^2) via Frobenius
  entrywise supabs bounds on dM/dx_k (entries are degree <= 1).

### 3b. Residual family (polynomial, so the pipeline stays certified)

    d(x) = delta * 0.25 * ( x2^2 ,  x1*x2 )

Certified residual constants as functions of delta, computed by bounds.py
on X = [-2,2]^2 (do not hand-evaluate; the coefficient bound gives
delta0(delta) and the Jacobian entries are linear, giving delta1(delta)).

### 3c. Experiment

Sweep delta over a log-spaced range bracketing the predicted budget
delta_star, where delta_star solves
eta = mu1*delta0(delta) + 2*b_upper*delta1(delta) (linear in delta, solve
in closed form from the certified constants). For each delta:
- measured worst raw value: max over the verification grid of
  lambda_max(C_true(x)) where C_true = C + sum_k (dM/dx_k) d_k
  + D^T M + M D, D = dd/dx (assemble polynomially, evaluate on the grid);
- theoretical bound: -eta + mu1*delta0(delta) + 2*b_upper*delta1(delta).

Figure 4 (fig_residual.pdf, single panel, same style as figs 1-3):
x-axis delta; two curves, measured worst value and theoretical bound;
horizontal line at 0; vertical line at delta_star; shade delta <=
delta_star. The measured curve must lie below the theoretical bound for
all delta (soundness check; if it does not, there is a bug, stop and fix).
Optional inset only if it stays legible: pairwise trajectory distance of
the true system at delta = 0.5*delta_star with the exp(-lambda'' t)
envelope, lambda'' from the theorem.

Write results/residual.json: eta, mu1, b_upper, delta_star, the sweep
table, and the measured/predicted values.

### 3d. Manuscript insertion

Insert as subsection V-D, after the grid-cost subsection, the following
skeleton (fill [XX:*] via fill_tex.py; keep the prose style rules of
CLAUDE.md if any sentence needs adjustment):

    \subsection{Residual robustness of the certified metric}
    We exercise Theorem~\ref{thm:residual} on the benchmark. The certified
    candidate attains uniform raw margin $\eta = [XX:eta]$ on $\X$, with
    certified constants $b \le [XX:b_upper]$ and $\mu_1 \le [XX:mu1]$
    computed by interval bounds on the polynomial entries of $M_\theta$ and
    its derivatives. We perturb the dynamics with the polynomial residual
    $d(x) = \delta \cdot 0.25\,(x_2^2,\; x_1 x_2)$, for which the certified
    residual constants are $\delta_0 = [XX:c_d0]\,\delta$ and
    $\delta_1 = [XX:c_d1]\,\delta$, so the theorem guarantees contraction of
    the true system for $\delta < \delta^\star = [XX:delta_star]$.
    Figure~\ref{fig:residual} compares, across a sweep of $\delta$, the
    measured worst value of the true contraction condition on the
    verification grid against the theoretical bound
    $-\eta + \mu_1\delta_0 + 2b\delta_1$. The measured curve remains below
    the bound throughout, the certificate survives up to
    $[XX:delta_emp_ratio]$ times the predicted budget, and the gap
    quantifies the conservatism of the norm-based proof of
    Theorem~\ref{thm:residual} on this benchmark.

    \begin{figure}[t]
    \centering
    \includegraphics[width=0.62\textwidth]{fig_residual.pdf}
    \caption{Residual robustness (Theorem~\ref{thm:residual}): measured
    worst value of the true contraction condition versus the theoretical
    bound as the residual scale $\delta$ grows. The shaded region is the
    certified budget $\delta < \delta^\star$.}
    \label{fig:residual}
    \end{figure}

---

## Task 4: manuscript consistency pass

- Replace the red footnote (the last remaining \todo) with exactly:

    \footnote{The oracle's Lipschitz constant is a certified upper bound
    computed by interval coefficient bounds, with box subdivision, on the
    polynomial entries of $\nabla_x C_\theta$ for the class
    \eqref{eq:class}; the implementation is part of the repository.}

- Also update the corresponding sentence in the experiments preamble that
  currently says the constant is "estimated by dense sampling with a
  safety factor of $1.5$": it must now state that all reported constants
  are certified, and may mention the sampled estimator only as the
  comparison baseline if a comparison is reported.
- Verify zero \todo markers remain: grep the tex.
- Compile twice; zero errors, zero undefined references; page render of
  the new figure page looks correct (use pdftoppm and inspect dimensions,
  fonts).
- Update the abstract ONLY if Task 2 changed a number quoted in it
  (one retraining round, three orders of magnitude); use JSON values.

## Task 5: run_all.py and final checks

run_all.py executes: bounds self-tests, three-variant experiment,
grid-cost sweep, residual experiment, fill_tex.py, double pdflatex,
and prints a PASS/FAIL summary per task. Commit only when all PASS.
