# Executed experiment configurations

The three configurations that define the reported experiments. They are historical YAML snapshots, not complete execution-state proof. In particular, the old RAPPPID hidden-size field differs from the executed 64-unit setting. For anonymity, absolute artifact
paths are replaced by `${SEQUENCE_BENCHMARK_ARTIFACT_ROOT}` so the files carry no
developer-specific location.

| File | Experiment | Cohort | Sequence ceiling |
|---|---|---|---|
| `e003_512.yaml` | E003 method-fidelity Core-4 | 512 residues | 512 |
| `e005_1024.yaml` | E005 primary replication | 1024 residues | 1024 |
| `fully_random_1024.yaml` | Two-endpoint Fully Random arm | 1024 residues | 1024 |

## What the configurations pin

* **Positives.** The frozen Gold positive backbone, unchanged across all six
  negative constructions. Only the negative pairs differ between constructions.
* **Splits.** Protein-disjoint train/validation/test. The E005 cohort reuses the
  independently verified E004 1024-residue panels.
* **Negative semantics.** Every Gold negative is declared
  `synthetic_unobserved_pair`. The configurations never call them experimental
  non-interactions.
* **Samplers.** Uniform, degree-matched, sequence-matched, ESM-matched, the
  Bernett 2026 global ILP sampler and the TPPNI configuration-model sampler, each
  with its upstream repository and revision recorded.
* **Model selection.** Checkpoints maximise matched-validation AUPRC; thresholds
  maximise validation MCC. Test adaptation, test checkpoint reselection and test
  threshold selection are all forbidden by the config.
* **Seeds.** 11, 22 and 33.

## What the configurations do not pin

The executed D-SCRIPT and TUnA adapters came from a SHA-bound historical
snapshot that is not redistributed; see `src/ppi_shift_release/models/README.md`
for the exact adaptation contract. The configurations record the model identity
and the length contract, not the adapter source.

## Reproducing a run

The configurations are documentation of what was executed, not a runnable
pipeline in this package. Running them requires the frozen panels, the ESM2-150M
weights and the trained checkpoints, the latter two of which are not redistributed; the 44 frozen panels are supplied. See
`docs/LIMITATIONS.md` for the exact boundary.
