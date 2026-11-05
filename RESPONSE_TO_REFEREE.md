# Response to the Referee

Manuscript: *Neurosymbolic Synthesis of Contraction Certificates*

## v1.0.1: related work and arXiv preparation

The introduction now cites Fitzsimmons and Liu (2024), distinguishing their
post-training SMT verification of continuous-time autonomous neural contraction
metrics from the present analysis of counterexample-driven retraining.
The README now describes the files actually retained in the released tree.
Release v1.0.1 includes a cleanly extracted and compiled arXiv source bundle.
The manuscript uses a single-spaced one-column layout with continuous line
numbers. Only the referee-mode presentation option changes; the mathematical
statements, scientific results, and original full-run evidence are preserved.
The current release citation is
https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.0.1.

## Final wording and archival release

The final manuscript qualifies abstract scale invariance by a scale-equivariant
oracle, restricts the margin comparison to stalled candidates, and identifies
the growth statistic as the certified Lipschitz bound for the normalized test.
It cites immutable release v1.0.0 at
https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.0.0.
The release preserves the exact verified reproduction ZIP and provides the
final line-numbered manuscript separately. An editorial rebuild manifest
records these wording and citation changes while retaining the original
full-run evidence and all scientific results without alteration.

## Follow-up revision: eight remaining issues

The subsequent line-numbered review identified additional mathematical,
scientific, and reproducibility issues. This revision addresses them as follows;
the earlier response below records the preceding revision.

1. Corollary 2 now assumes a fixed shape strictly negative over the domain,
   a positive scale-equivariant Lipschitz bound, scale closure, and an initially
   positive certifiability gap. The proof keeps every augmented dataset inside
   that negative domain and constructs increasing scales explicitly.
2. The manuscript defines variational displacements and distinguishes their
   decay from finite pairwise convergence. The latter requires an admissible
   connecting curve whose full flow stays in the domain; convex forward
   invariance is a stated sufficient condition. The polynomial box supplies
   no unconditional finite-trajectory guarantee.
3. The inaccurate claim that the public prototype contains this revision is
   removed. A checked local reviewer archive includes source, dependencies,
   results, figures, paper, and provenance. Packaging verifies hashes against
   a complete passing pipeline. The final archival release above supplies the
   immutable public identifier.
4. True uniform slack and its computable certified lower bound have distinct
   notation. The grid-sufficiency proof uses the Lipschitz estimate directly,
   and the grid-size formula handles zero cases explicitly.
5. Generated protocol metadata and the manuscript table state outcome
   classifications, per-study round caps, and training budgets. The labels
   remain finite-budget observations rather than asymptotic theorems.
6. The neural comparison describes the observed parametrization/oracle
   combination. It no longer isolates interval-bound conservatism as the cause
   when learned functions and optimization settings also change.
7. Successful neural runs export outward-certified metric bounds and the
   resulting conservative overshoot and improved differential-rate bounds.
   The result validators reject missing or inconsistent quantitative outputs.
8. The stall lemma now claims the existence of a zero-subgradient selection
   with zero optimizer state. It does not claim that every optimizer or
   boundary subgradient choice remains fixed.

Validation evidence, including arithmetic acceptance, regression checks,
fresh scientific outputs, and two LaTeX passes, is recorded in the generated
run manifest. The manuscript retains continuous reviewer line numbers.

## Earlier revision response

We thank the referee for a careful reading and for verifying the mathematics in
detail. The report identified one fatal issue (an oracle that was not sound as
implemented), a gap in the positioning against prior art, and a thin
instantiation relative to the framing. We have addressed all three. The oracle
now uses analytic interval coefficient bounds with box subdivision, and the
reported experiments use these bounds. A numerical-implementation paragraph
states the stored-parameter and arithmetic assumptions. Exact rational
coefficient construction and outward-rounded interval bounds and grid checks
now account for rounding error. MPFR supplies directed transcendental
enclosures through gmpy2; dense sampling remains a consistency check. Experiment
outputs supply the numerical macros in Section V.

One disclosure up front, because it changes Section V. Replacing the sampled
Lipschitz estimate with the certified bound (Major 1) changed the experimental
picture, and we report the new picture honestly rather than preserve the
original single-trajectory story. The analytic bound can exceed the sampled estimate substantially, which makes
the margin the loop must clear larger. Two consequences follow. First, loop-outcome claims now use statistics over 20
seeds on one machine. The residual experiment remains a single-candidate
illustration of the theorem. Second, the "scale chase" no
longer appears as an empirical loop divergence: with a certifiable shape in the
class the gradient learner reshapes rather than inflating scale, exactly the
degeneracy-not-inevitability our Corollary 2 discussion describes. We kept the
scale chase as a distinct theoretical mechanism (Corollary 2) and report the divergence
that does occur, which is driven by the metric's shape through its Lipschitz
constant when the coupled margin is infeasible for the grid. Section V now
presents three regimes, stall, divergence, and certification, over 20 seeds.

Section and result references below are to the revised manuscript. Major-comment
headings retain the referee's original numbering and result names; the new
coefficient, constant, and grid-sufficiency lemmas are Lemmas 1--3, the scale
lemma is Lemma 4, and the stall lemma is Lemma 5.

## Major comments

**1. Soundness of the oracle.** Fixed. A new Section III subsection, *Certified
constants for the polynomial class*, states Lemma 1 (coefficient bound with box
subdivision, monotone in depth), Lemma 2 (certified Lipschitz constants for both
the raw test and the normalized quotient, including the quotient constant the
oracle consumes, together with certified $a$, $b$, and $\mu_1$), and Lemma 3
(grid sufficiency). The quotient constant the referee noted was missing is now
explicit: it uses $\sup|\tilde g|$, the trace's Lipschitz constant, and the exact
lower bound $n\alpha$. The standing hypothesis is strengthened from $f \in C^1$
to $f \in C^2$, a sufficient but stronger condition than local Lipschitz
continuity of $A$, with a one-line justification. The Section V footnote no longer mentions sampling or a safety
factor, and the editorial placeholder is removed. All experiments were rerun with
the certified bound; the implementation and self-tests are in the repository
(`bounds.py`, acceptance tests `test_1b`, `test_1c_lower_bound`,
`test_1c_monotone`).

**2. Prior art and the novelty claim.** Fixed in Section I. (a) We cite
Aylward, Parrilo, and Slotine (2008) for SOS synthesis of contraction metrics,
acknowledge that the polynomial benchmark is within SOS reach, and state that we
use it because its certified answer is known, which isolates the loop's failure
modes. We also add a damped-pendulum benchmark with trigonometric dynamics
(Section V.E), using direct interval enclosures. Standard polynomial SOS would
require an approximation with a controlled remainder or a reformulation; we do
not claim that all SOS approaches are excluded. (b) We cite Giesl
(2019) for verified numerical construction by meshfree collocation. (c) We cite
Boffi et al. (CoRL 2020, proceedings published in 2021) for data-driven certificates. (d) We now characterize
reference [12] (Harapanahalli, Coogan, Davydov) precisely: it jointly trains a
neural controller and a neural metric with an interval bound-propagation check
inside training, it targets the controlled problem, and it does not analyze the
scale-invariance obstruction or the stall. We also cite Li, Zhong, Hu, and Zhang (L4DC 2025), who train discrete-time
contraction metrics using projected-gradient counterexamples and verify them
with alpha-beta-CROWN. Their local-distance condition differs from the
continuous-time differential matrix inequality studied here. We remove the
blanket priority claim. Our contribution is the continuous-time autonomous
CEGIS-C analysis of scale degeneracy, fixed-margin stalls, and verified-slack
guarantees, together with numerical illustrations.

**3. Neurosymbolic framing versus instantiation.** Addressed. We delivered the
interval oracle the referee called mechanical (Major 1), and we tempered the
terminology: the abstract now says "sound verification oracle" with certified
interval bounds, and "learner" in place of "neural learner." A new paragraph in
Section I states that the framework is agnostic to the learner and oracle
architectures. It identifies both instantiations already present: a metric
linear in fixed features with a coefficient-bound oracle, and a neural metric
with interval bound propagation. We include the latter (Section V.F): a
one-hidden-layer tanh network for the Cholesky factor, certified by hand-rolled
interval bound propagation. As the referee anticipated, the clean coarse-grid
certification does not survive the change; the certificate does survive, but only
at a substantially finer grid, and we report that cost.

**4. Corollary 2.** Fixed. (a) The hypothesis $g_\theta < 0$ on the data is now
stated. (b) A paragraph after the corollary notes that the implemented class with
fixed $\alpha$ is not exactly scale-closed, so the mechanism is asymptotic
($\|L\|$ inflation until the floor is negligible), and states the theory-
experiment link at that honesty. (c) "Unbounded growth" is restated as the
existence of learner responses satisfying every requirement with
$L(\theta_k) \to \infty$. The certifiability gap is now defined (Definition of the
certifiability gap) where first used, with the note that it is scale-covariant on
the raw test and scale-invariant on the normalized test.

**5. Lemma 2 and Algorithm 1.** Fixed. The stall result, now Lemma 5, is restated: $\theta$ is a
stationary point of the empirical hinge loss for every dataset, and any
(sub)gradient method initialized at $\theta$ admits $\theta$ as a fixed point.
Algorithm 1's learner line is rewritten as a finite-step subgradient update
(Adam), with a sentence explaining that we analyze the finite-step learner rather
than an exact argmin because the argmin is non-unique under the stall.

**6. Progress and completeness.** Addressed. Lemma 3 formalizes the sufficiency
direction the referee identified: true uniform slack exceeding $L_h\tau$ implies
the oracle certifies. The opening of Section V.C is corrected from the false "if
and only if" to sufficiency plus the converse. Definition 1 now says the grid
oracle is sound but not complete, and a remark after Proposition 2 states
soundness without completeness and poses relative completeness (finite
certification under realizability with margin and exact minimization) as an open
problem, with Lemma 3 as its sufficiency half. A full relative-completeness
theorem remains future work.

**7. Experimental scope and inconsistencies.** (a) We add the
residual-robustness study of Theorem 2 (Section V.D, Figure 4), a
pendulum benchmark without a known certificate supplied by construction
(Section V.E, Figure 5), and a neural metric (Section V.F, Figure 6). We remove
the unsupported claim that the pendulum class has no certificate at rate 0.4.
The pendulum and neural studies report normalized coupled-margin runs, not a
fixed-margin stall comparison. The neural study also changes the initialization
and optimizer settings, so it does not isolate expressiveness alone. (b) The
Figure 3 inconsistency is resolved: every number is now read from
`results/gridcost.json` through `fill_tex.py`, and the extrapolation uses the
median sampled slack-to-Lipschitz ratio over the certified seeds, stated in the
text. Sampled slack is not a certified lower bound on uniform slack. The
higher-dimensional grid counts hold that measured two-dimensional ratio fixed
and are cost extrapolations, not higher-dimensional experiments or certified
minimum grid requirements.
(c) Figure 1 (gap traces) is redesigned: the normalized certifiability gap for all regimes on
one scale-invariant axis, on a symlog scale so certification appears as a
crossing below zero, with per-seed lines and per-regime medians and counts in the
legend. (d) Section V.A now reports the certified class bounds $a$ and $b$, the
initial dataset size, Adam steps per round, seeds, coupling factor, both grids,
and the covering radius.

**8. Interface between loop and theorems.** Fixed. The remark after Theorem 1 is
rewritten cleanly via $\tr M_\theta \ge na$: if $\tilde g_\theta \le -\eta_n$
then $g_\theta \le -na\,\eta_n$, so the uniform raw margin $\eta = na\,\eta_n$ is
delivered to Theorems 1 and 2, and Algorithm 1 returns the normalized margin $s$ and class bounds $a,b$
needed to compute it, with $\eta = nas$.
We clarify that the surrogate's Lipschitz bound alone does not bound the
residual Jacobian. The theorem needs direct derivative-error information,
separate bounds on the true and learned vector fields, or a statistical
derivative-error bound under explicit model assumptions. A magnitude bound
alone does not justify setting $\delta_1=0$. A remark in
Section V.A notes that the box is not forward invariant and that certificates
should be read on a forward-invariant subset.

**9. Necessity of normalization.** Fixed. Lemma 4 identifies a scale
degeneracy; it does not prove that normalization is necessary for every learner
to make progress. We state this distinction, note that the generalized-eigenvalue
normalization is equally scale-invariant, and justify the trace choice by the
tractability of its denominator and lower bound $n\alpha$. We remove broad
necessity and minimality claims. Coupling sets the next training margin using
the current candidate's bound; retraining can change that bound, so coupling
does not guarantee convergence.

## Minor comments

1. $\eta$ and $b$ are now defined in the abstract, and "sound symbolic oracle" is
   tempered (per Major 3).
2. Definition 1 now states the oracle is sound but incomplete and links to the
   relative-completeness remark.
3. Lemma 4 now carries the hypothesis that the oracle constant is produced by a
   scale-equivariant bounding procedure, which our coefficient bound satisfies.
4. The conclusion is aligned: "a scale-chase degeneracy," not "divergent."
5. Section V.A states that perturbed counterexamples are projected into $\X$ by
   clipping.
6. Section V.A reports the grids, hence the grid evaluation count
   $N_{\mathrm{ax}}^2$ per oracle call. The reproduction report records stage
   runtimes and the execution environment. The separate
   `timing_polynomial.json`, `timing_pendulum.json`, and `timing_neural.json`
   outputs record learner and oracle times and grid evaluation counts per round;
   oracle time includes bound computation. We remove the unsupported per-call
   timing ranges and distinguish measured timing from deterministic results.
7. We checked the Aylward, Giesl, and Boffi pagination against primary sources.
   Harapanahalli, Coogan, and Davydov now carry their published IEEE Control
   Systems Letters citation (vol. 10, pp. 1615--1620, 2026), DOI, and arXiv ID.
   We also add the published L4DC 2025 citation for Li et al. The literature
   check does not establish exhaustive priority, and we remove the broad first
   claim.
8. The draft artifacts are removed: the watermark, the working-draft headers, the
   placeholder, and the `\todo` macro.
9. Proposition 2's proof now notes that a strict grid margin upgrades the
   certificate to the improved rate of Theorem 1.
10. Section V.A is corrected: the class contains certifiable metrics at the target
    rate ($\alpha I + c\,J_\varphi^\top J_\varphi$ for $c$ large), while the fixed
    floor keeps $M^\star$ itself outside the family.

## Note on the two structural enhancements

The referee's structural requests were a benchmark beyond the original
polynomial representation and a neural metric. We added both. The non-polynomial pendulum
(Section V.E) is certified by the polynomial oracle extended to enclose sin and
cos as bounded atoms refined by subdivision; the metric stays polynomial, so no
remainder algebra is needed. The neural metric (Section V.F) is a one-hidden-layer
tanh network certified by hand-rolled interval bound propagation through the exact
computation of the contraction matrix and its partials, with tanh and its first
two derivatives enclosed with outward endpoints. Exact coefficient algebra
and directed interval operations preserve these enclosures through bound
construction and grid checks. The verifier rounds Lipschitz margins upward
and uniform slack and sufficient residual budgets downward. Dense sampling
checks the implementations; positive answers require interval verification. Both studies report
certification and divergence under the normalized coupled-margin loop, without
claiming to test the fixed-margin stall. The
neural study finds fewer coarse-grid certifications for the tested neural
parametrization and oracle. The functions and optimizer settings also change,
so this experiment does not isolate interval-bound conservatism as the cause.


## Final consistency notes

The residual sweep stays negative at its largest sampled scale, four times the
theorem's budget in the stored study. The text now treats this as a sweep
endpoint, not a certified extension of the budget or an observed failure
threshold. The conclusion separates theoretical scale chasing from the
shape-dependent divergence observed in normalized runs. Outcome labels describe
finite-budget behavior, not proof of exact fixed points or unbounded trajectories.

Bibliographic checks used the [Aylward author publication list](https://www.mit.edu/~parrilo/pubs/index.html),
the [Giesl publisher record](https://epubs.siam.org/doi/abs/10.1137/18M1220182),
the [Boffi proceedings record](https://proceedings.mlr.press/v155/boffi21a.html),
the [Harapanahalli author publication list](https://www.coogan.ece.gatech.edu/pubs.html)
and [paper](https://arxiv.org/html/2603.28011v2), and the
[Li proceedings record](https://proceedings.mlr.press/v283/li25e.html) and
[paper](https://raw.githubusercontent.com/mlresearch/v283/main/assets/li25e/li25e.pdf).

## Review format and rounding revision

The rebuilt manuscript uses the single-column IEEE review draft with continuous
line numbers, including displayed equations. The numerical-implementation
paragraph describes exact binary64 input semantics, exact polynomial
coefficient algebra, outward elementary operations, MPFR transcendental
enclosures, the covering radius of the actual stored grid, and fail-closed
arithmetic checks. Ordinary floating-point screening only rejects candidates.
All positive outputs require an outward-rounded grid certificate; the
reproduction records the arithmetic backend and its acceptance checks.
