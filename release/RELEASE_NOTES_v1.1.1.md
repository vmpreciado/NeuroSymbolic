This immutable release supplies a layout and navigation update to the verified
preprint **Neurosymbolic Synthesis of Contraction Metrics with Certified
Verification**. It preserves the scientific claims, abstract, formal mathematical
statements and proofs, numerical results, and original twelve-stage run evidence
from v1.1.0.

## Changes

- Compacts the table layout so the adaptive-verification and fitted-surrogate
  results no longer occupy a sparsely filled isolated float page.
- Updates the manuscript and citation metadata to identify this version.
- Provides current paper, source, reproduction, and independent-review assets;
  a default-branch navigation README directs readers to the versioned current
  project while preserving the historical code on that branch. Its exact text
  is included as `release/DEFAULT_BRANCH_README.md`.

No learning or experiment producer was rerun for this layout update. The build
revalidates saved scientific results and adaptive certificates, confirms that the
abstract and formal mathematics are unchanged from the immutable v1.1.0 source,
and tests compilation of the current paper and extracted arXiv sources.

## Scientific evidence and provenance

`neurosymbolic-reproduction-scientific-verified.zip` remains the exact original
passing twelve-stage scientific snapshot, SHA256
`c1cdea042cca4f5495a3ed025174570bb68e494df2695641af2e97d91437eb39`.
Its run manifest, scientific programs, saved results, generated numbers and tables,
figures, and stage logs are preserved unchanged. The original archive's manuscript
predates the current editorial and layout updates.

`editorial_v1.1.1.json` records the current input/output hashes and the relation to
both the unchanged scientific snapshot and the preceding immutable v1.1.0 source.
The current reproduction ZIP contains runnable source and evidence, the final
unnumbered manuscript, the original snapshot, and a standalone integrity verifier.
The source archive contains only the thirteen flat TeX dependencies. The Claude
package includes the same current paper, code and evidence, with a neutral prompt
for independent scientific review.

## Assets

- `neurosymbolic-paper-v1.1.1.pdf`: authoritative paper without reviewer line numbers.
- `neurosymbolic-arxiv-v1.1.1.tar.gz`: minimal clean-tested arXiv source bundle.
- `neurosymbolic-reproduction-v1.1.1.zip`: complete current runnable reproduction.
- `neurosymbolic-reproduction-scientific-verified.zip`: unchanged original scientific run.
- `neurosymbolic-claude-deep-review-v1.1.1.zip`: independent-review package.
- Abstract, citation metadata, editorial provenance, and checksum inventory.

`SHA256SUMS` lists the published asset hashes. The reproduction's
`VERIFY_CURRENT_REPRODUCTION.py` and the review package's `VERIFY_PACKAGE.py`
validate their extracted inventories and provenance without third-party Python
dependencies. These checks establish integrity, not mathematical correctness.

Earlier immutable releases, including v1.1.0, remain unchanged. The scientific
limitations are also unchanged: idealized rather than practical termination
assumptions, finite training/verification budgets, no matched external-method
performance comparison, and known-truth residual certification in the surrogate
experiment. This release prepares the paper for arXiv and does not submit it.

Citation: Victor M. Preciado, *Neurosymbolic Synthesis of Contraction Metrics with
Certified Verification: Verified Reproduction and Submission Artifacts*, v1.1.1,
GitHub release, 2026. https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.1.1
