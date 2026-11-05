# Verified reproduction snapshot and final review manuscript

Release **v1.0.1** archives the verified scientific reproduction for
*Neurosymbolic Synthesis of Contraction Certificates* and the final
line-numbered preprint, with a tested arXiv source bundle. The immutable
v1.0.0 release remains unchanged.

## Scientific snapshot

`neurosymbolic-reproduction.zip` is the exact, unchanged archive from the
passing eight-stage run. It includes the source, pinned dependency
specifications, tests, scientific results, all six figures, the manuscript at
that run, stage logs, and provenance. Its SHA256 is:

```text
c67b4247793c591125dfd71c1d86776a21b95ba1a59d116126c51d29cd934070
```

## Final manuscript

Use the separately attached `nsp_contraction_paper_ieee.pdf` for the current
preprint. It adds Fitzsimmons and Liu (2024) to the related-work discussion,
distinguishes their post-training SMT verification from the current synthesis
analysis, and cites this release. The README removes references to historical
files absent from the published tree. The manuscript uses single spacing and
retains continuous line numbers; only the double-spaced referee-mode option
was removed from the presentation settings.

The verified ZIP deliberately preserves its earlier manuscript byte-for-byte.
The attached final TeX, `numbers.tex`, and the tag's paper sources include the
editorial updates. `editorial_rebuild.json` and `editorial_build.log` document
the new build and verify that scientific source, data, figures, and original
full-run evidence were reused unchanged. The editorial build validated the
saved results, regenerated numerical macros, and completed two LaTeX passes
with no errors, unresolved references, or warnings. It did not rerun learning
or the scientific experiments.

## Integrity and reproduction

`neurosymbolic-arxiv-v1.0.1.tar.gz` contains exactly eight manuscript inputs:
the main TeX source, generated `numbers.tex`, and six figure PDFs, all at the
top level. The bibliography is embedded in the TeX. The archive excludes the
compiled manuscript, logs, code, scientific data, and nested archives. The
clean extraction and two-pass PDFLaTeX check are documented in
`arxiv_bundle.json`. It is a prepared source submission, not an arXiv upload.

GitHub release immutability protects the published assets and version tag.
`SHA256SUMS` lists the attached asset checksums. The tag's README documents
both full scientific reproduction and a paper-only rebuild. Installing the
specified Python packages and a suitable TeX distribution is required.

Citation: Victor M. Preciado, *Neurosymbolic Synthesis of Contraction
Certificates: Verified Reproduction Snapshot and Review Manuscript*, version
v1.0.1, GitHub release, 2026.

https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.0.1
