This release resolves the descriptive-rounding discrepancy and explains the polynomial constant-metric choice and proof margin. The paper retains **Neurosymbolic** in the title and has no reviewer line numbers.

The repeated polynomial median upper-bound statistic is now displayed as `4.25` in both the prose and Table III. Descriptive summaries are explicitly rounded to nearest; individual certificate upper bounds retain upward rounding, and certified lower guarantees retain downward rounding. All verification calculations still use saved full-precision values.

For the polynomial benchmark, `P = diag(p,1)` strictly satisfies the contraction inequality when `p > B^2/(1-lambda)^2`. At `B = 2` and the exact decimal rate `lambda = 2/5`, the threshold is `100/9`, explaining the simple integer choice `p = 12`. The release adds an exact rational derivation record and an independent boundary test. The raw proof margin `1/20` is chosen independently of the normalized learner margin; separate shifted endpoint checks prove it. The original endpoint certificates, decimal/binary64 rate checks, and all learning protocols are preserved.

A fresh fifteen-stage pipeline regenerates every experiment, verifies both constant-metric baselines, rebuilds the paper, and compiles the extracted arXiv source. The complete reproduction and Claude review package contain source and artifact hashes, numerical environment records, all seed outcomes, positive proof records, and stage logs. The Claude prompt now explicitly asks reviewers to examine the clarified rounding and baseline rationale.

The scientific conclusions and limitations are unchanged. The matched polynomial experiments did not demonstrate a certification-count advantage from normalization or margin coupling. Constant metrics certify the polynomial benchmark and smallest swing-model box, so those results do not require neural state dependence. The analytical baselines are post hoc fixed candidates, without optimized search or matched timing comparisons.

Downloads include the paper, arXiv source archive, complete runnable reproduction, Claude review package, both baseline proof JSON files, abstract and citation metadata, and SHA256SUMS. The paper and CITATION.cff cite this exact release. Earlier immutable releases, including v1.2.1, remain unchanged. No arXiv submission is performed by this release.

Scientific source freeze: `a821d6c64441004a1ef63f8e1b89635d966aac13`. The release tag also includes the regenerated artifacts; the full-run manifest records source stability throughout the pipeline.

Validation: all 15 pipeline stages and 164 tests passed. The fresh run took 1,945.5 seconds, with all 63 source hashes unchanged and 45 artifacts recorded. Independent comparison found all 16 existing scientific JSON outputs unchanged from v1.2.1 after explicitly excluding runtime metadata and the arithmetic-check timestamp; all 180 application candidates, outcomes, positive proof covers, and guarantees agree. The added polynomial-choice explanation passed independent exact matrix and threshold checks.

The paper has 30 pages, no reviewer line numbers, and an unchanged 1,900-character abstract. The 17-file arXiv archive passed an independent three-pass compilation with zero warnings; every page's text and decoded content stream matches the delivered PDF. Independent audits checked all 125 reproduction files and 136 review payloads.

All 30 rendered pages passed visual inspection, including the revised baseline paragraph, matching median displays, release citation, and final references.
