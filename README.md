# Neurosymbolic Synthesis of Contraction Metrics with Certified Verification

**Use [immutable release v1.1.1](https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.1.1) for the current paper and reproducible project.** The default branch retains historical prototype code. Obtain the versioned release assets or check out the release tag before reproducing the paper.

The paper studies counterexample-guided synthesis of contraction metrics for continuous-time autonomous systems, with outward-rounded numerical certification. Its main contribution combines failure-mechanism analysis, computable certificates, and reproducible diagnostics. The matched experiments did not demonstrate a certification-count advantage from normalization or margin coupling within the tested budgets.

Download the current materials:

- [Paper without reviewer line numbers](https://github.com/vmpreciado/NeuroSymbolic/releases/download/v1.1.1/neurosymbolic-paper-v1.1.1.pdf)
- [Tested arXiv source bundle](https://github.com/vmpreciado/NeuroSymbolic/releases/download/v1.1.1/neurosymbolic-arxiv-v1.1.1.tar.gz)
- [Complete verified reproduction](https://github.com/vmpreciado/NeuroSymbolic/releases/download/v1.1.1/neurosymbolic-reproduction-v1.1.1.zip)
- [Claude independent-review package](https://github.com/vmpreciado/NeuroSymbolic/releases/download/v1.1.1/neurosymbolic-claude-deep-review-v1.1.1.zip)
- [Release checksums](https://github.com/vmpreciado/NeuroSymbolic/releases/download/v1.1.1/SHA256SUMS)

For source-based reproduction, select the version first:

```sh
git clone https://github.com/vmpreciado/NeuroSymbolic.git
cd NeuroSymbolic
git checkout v1.1.1
```

Then follow the [versioned reproduction instructions](https://github.com/vmpreciado/NeuroSymbolic/blob/v1.1.1/README.md), including the locked Python requirements, TeX dependencies, verification commands, and distinction between a fresh scientific run and an editorial rebuild. The reproduction ZIP is a runnable alternative and includes its own standalone integrity verifier.

Version v1.1.1 tightens the paper layout and improves project navigation. The scientific implementation, saved results, and original passing twelve-stage evidence are preserved, with explicit provenance for the rebuilt manuscript and packages.

Use the [versioned citation metadata](https://github.com/vmpreciado/NeuroSymbolic/blob/v1.1.1/CITATION.cff) when citing these materials. Earlier immutable releases [v1.1.0](https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.1.0) and [v1.0.1](https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.0.1) remain unchanged. The release prepares the paper for arXiv; it does not itself post an arXiv submission.
