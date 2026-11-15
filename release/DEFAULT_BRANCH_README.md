# Neurosymbolic Synthesis of Contraction Metrics with Certified Verification

**Use [immutable release v1.2.2](https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.2.2) for the current paper and reproducible project.** The default branch retains historical prototype code. Obtain the versioned release assets or check out the release tag before reproducing the paper.

The paper studies counterexample-guided synthesis of contraction metrics for continuous-time autonomous systems, with outward-rounded numerical certification. Its main contribution combines failure-mechanism analysis, computable certificates, and reproducible diagnostics. The matched experiments did not demonstrate a certification-count advantage from normalization or margin coupling within the tested budgets.

Download the current materials:

- [Paper without reviewer line numbers](https://github.com/vmpreciado/NeuroSymbolic/releases/download/v1.2.2/neurosymbolic-paper-v1.2.2.pdf)
- [Tested arXiv source bundle](https://github.com/vmpreciado/NeuroSymbolic/releases/download/v1.2.2/neurosymbolic-arxiv-v1.2.2.tar.gz)
- [Complete verified reproduction](https://github.com/vmpreciado/NeuroSymbolic/releases/download/v1.2.2/neurosymbolic-reproduction-v1.2.2.zip)
- [Claude independent-review package](https://github.com/vmpreciado/NeuroSymbolic/releases/download/v1.2.2/neurosymbolic-claude-deep-review-v1.2.2.zip)
- [Release checksums](https://github.com/vmpreciado/NeuroSymbolic/releases/download/v1.2.2/SHA256SUMS)

For source-based reproduction, select the version first:

```sh
git clone https://github.com/vmpreciado/NeuroSymbolic.git
cd NeuroSymbolic
git checkout v1.2.2
```

Then follow the [versioned reproduction instructions](https://github.com/vmpreciado/NeuroSymbolic/blob/v1.2.2/README.md), including the locked Python requirements, TeX dependencies, and verification commands. The reproduction ZIP is a runnable alternative. The Claude review package includes a standalone integrity verifier.

Version v1.2.2 harmonizes descriptive rounding and explains the polynomial constant-metric choice and its independently selected proof margin. Its fresh fifteen-stage pipeline includes every prior experiment, the physical swing-model application, both constant-metric baselines, and clean manuscript/source builds. The polynomial benchmark and smallest swing-model box admit fixed constant metrics, so those results do not establish a need for neural state dependence. The learning comparisons do not establish general superiority over existing tools.

Use the [versioned citation metadata](https://github.com/vmpreciado/NeuroSymbolic/blob/v1.2.2/CITATION.cff) when citing these materials. Earlier immutable releases [v1.2.1](https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.2.1), [v1.2.0](https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.2.0), [v1.1.1](https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.1.1), [v1.1.0](https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.1.0) and [v1.0.1](https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.0.1) remain unchanged. The release prepares the paper for arXiv; it does not itself post an arXiv submission.
