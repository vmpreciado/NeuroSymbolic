# Verified reproduction snapshot and final review manuscript

Release **v1.0.0** archives the verified scientific reproduction for
*Neurosymbolic Synthesis of Contraction Certificates* and the final
line-numbered review manuscript.

## Scientific snapshot

`neurosymbolic-reproduction.zip` is the exact, unchanged archive from the
passing eight-stage run. It includes the source, pinned dependency
specifications, tests, scientific results, all six figures, the manuscript at
that run, stage logs, and provenance. Its SHA256 is:

```text
c67b4247793c591125dfd71c1d86776a21b95ba1a59d116126c51d29cd934070
```

## Final manuscript

Use the separately attached `nsp_contraction_paper_ieee.pdf` for the final
review draft. It corrects the abstract's scale-equivariance qualification,
restricts the margin comparison to stalled candidates, identifies the growth
statistic as the certified bound for the normalized test, and cites this
release. It retains continuous reviewer line numbers.

The verified ZIP deliberately preserves its earlier manuscript byte-for-byte.
The attached final TeX, `numbers.tex`, and the tag's paper sources include the
editorial updates. `editorial_rebuild.json` and `editorial_build.log` document
the new build and verify that scientific source, data, figures, and original
full-run evidence were reused unchanged. The editorial build validated the
saved results, regenerated numerical macros, and completed two LaTeX passes
with no errors, unresolved references, or warnings. It did not rerun learning
or the scientific experiments.

## Integrity and reproduction

GitHub release immutability protects the published assets and version tag.
`SHA256SUMS` lists the attached asset checksums. The tag's README documents
both full scientific reproduction and a paper-only rebuild. Installing the
specified Python packages and a suitable TeX distribution is required.

Citation: Victor M. Preciado, *Neurosymbolic Synthesis of Contraction
Certificates: Verified Reproduction Snapshot and Review Manuscript*, version
v1.0.0, GitHub release, 2026.

https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.0.0
