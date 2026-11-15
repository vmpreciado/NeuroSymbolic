This release adds a physical two-machine swing-model application, a controlled comparison with an independently implemented adaptation of the Fitzsimmons–Liu PDI objective, and a verified analytical constant-metric baseline. The manuscript retains the title **Neurosymbolic Synthesis of Contraction Metrics with Certified Verification** and has no reviewer line numbers.

The analytical matrix `P = [[1, 1/2], [1/2, 2]]` certifies the smallest application box at the same requested rate and strict margin as the neural comparison. Directed MPFR enclosures and exact rational matrix checks cover both exact-decimal and stored binary64 targets. The candidate has rigorous whole-box violations on the two larger domains. These failures concern this fixed matrix and do not exclude other constant metrics or contraction on the interior invariant energy subsets. The baseline was added after the neural experiment and has no training seeds or matched timing claim.

The application retains all 180 paired neural runs. Its small-box primary counts are 20/20 for the CEGIS hinge learner, 18/20 for the adapted PDI objective, and 20/20 for the fixed-data hinge control. Larger primary boxes have no certificates. An explicitly exploratory verification-budget increase certifies two frozen middle-box CEGIS candidates without retraining. The evidence does not establish general superiority, a need for neural metrics on the smallest box, or an advantage over the original complete LyZNet/SMT workflow. The earlier polynomial ablation still reports no demonstrated certification-count benefit from normalization or margin coupling.

A fresh fourteen-stage reproduction regenerates the scientific results, verifies the analytical baseline and proof records, rebuilds the paper, and compiles the extracted arXiv source in a clean directory. The archives retain the numerical environment, source and artifact hashes, all seed outcomes, and complete stage logs. The independent-review package includes a standalone integrity/provenance verifier; checksums and passing tests do not substitute for mathematical review.

Assets:

- `neurosymbolic-paper-v1.2.0.pdf`: current paper without reviewer line numbers.
- `neurosymbolic-arxiv-v1.2.0.tar.gz`: minimal, clean-build-tested arXiv source.
- `neurosymbolic-reproduction-v1.2.0.zip`: complete verified runnable snapshot.
- `neurosymbolic-claude-deep-review-v1.2.0.zip`: paper, source, reproduction, evidence, and independent review prompt.
- `SHA256SUMS`: checksums for the uploaded artifacts and metadata.

The manuscript and CITATION.cff cite this exact release. Previous immutable releases remain unchanged. This release does not submit the paper to arXiv.

Scientific source freeze: `5b67589cf4e8e9660aac05b5d2098d62e4b0100a`. The release tag also records the generated outputs; the full-run manifest verifies that all scientific source hashes remained unchanged.

Validation: 153 tests and all 14 pipeline stages passed. The paper has 30 pages, no reviewer line numbers, a 1,883-character abstract, and zero LaTeX warnings. The extracted 16-file arXiv source passed a clean three-pass build. Independent package audit verified all 30 page content streams against that build.
