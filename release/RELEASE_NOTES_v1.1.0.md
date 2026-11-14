This immutable release archives the current verified reproduction and the
unnumbered preprint **Neurosymbolic Synthesis of Contraction Metrics with Certified
Verification**. It incorporates the external referee revision, the Bullo/Jaffe/
Davydov literature comparisons, and the final novelty review.

## Manuscript changes

- Distinguishes dual control-contraction-metric formulations from direct joint
  controller-and-primal-metric learning.
- Identifies failure-mechanism analysis, computable outward-rounded certificates,
  and reproducible diagnostics as the principal contribution, with established
  sampling and CEGIS arguments as building blocks.
- Compares Davydov's partition-certificate training objective with the sampled
  hinge loss and diagnostic-point augmentation used here.
- Cites this release for the current paper and reproduction. The abstract retains
  the matched experiments' lack of a demonstrated certification-count advantage
  from normalization or margin coupling. The title retains Neurosymbolic, with
  the neural and symbolic roles defined explicitly.

## Scientific evidence and provenance

The verified scientific snapshot passed all twelve declared pipeline stages,
including controlled ablations, adaptive three-dimensional verification,
known-truth fitted-surrogate transfer, complete certificate reporting, and a
clean arXiv source compilation. Its exact ZIP is preserved unchanged as
`neurosymbolic-reproduction-scientific-verified.zip`, SHA256
`c1cdea042cca4f5495a3ed025174570bb68e494df2695641af2e97d91437eb39`.

The final manuscript is an editorial rebuild of those scientific outputs.
Saved results and adaptive certificates were revalidated, numerical inputs were
regenerated, and both the main and extracted arXiv sources compiled cleanly.
The final release does not claim that training or experiments were run again.
The original run manifest and stage logs remain byte-for-byte intact.

## Assets

- `neurosymbolic-paper-v1.1.0.pdf`: authoritative paper without reviewer line numbers.
- `neurosymbolic-arxiv-v1.1.0.tar.gz`: minimal tested TeX source bundle.
- `neurosymbolic-reproduction-v1.1.0.zip`: complete runnable current reproduction,
  with the final paper, saved evidence, explicit editorial provenance, the original
  scientific archive, and a standalone integrity verifier.
- `neurosymbolic-reproduction-scientific-verified.zip`: exact original scientific run.
- `neurosymbolic-claude-deep-review-v1.1.0.zip`: independent-review package with
  current paper/source, code, evidence, and a neutral review prompt.
- Abstract, citation metadata, editorial provenance, and checksum inventory.

`SHA256SUMS` lists the published asset hashes. The reproduction ZIP's
`VERIFY_CURRENT_REPRODUCTION.py` and the Claude package's `VERIFY_PACKAGE.py`
check their extracted contents without third-party dependencies. These checks
establish integrity and provenance, not mathematical correctness.

The main limitations remain explicit: conditional idealized termination, finite
training and verification budgets, no matched external-method performance
comparison, and known-truth residual certification in the surrogate experiment.
The earlier immutable v1.0.0 and v1.0.1 releases are unchanged. This release prepares
the paper for arXiv; it does not submit or post the paper there.

Citation: Victor M. Preciado, *Neurosymbolic Synthesis of Contraction Metrics with
Certified Verification: Verified Reproduction and Submission Artifacts*, v1.1.0,
GitHub release, 2026. https://github.com/vmpreciado/NeuroSymbolic/releases/tag/v1.1.0
