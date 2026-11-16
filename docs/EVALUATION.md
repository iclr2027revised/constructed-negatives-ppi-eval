# Evaluation conventions

## Pointwise Gold metrics

The 1,296 released cells in `results/pointwise/` comprise 432 `E005_core` original unshuffled 1024-residue Gold predictions, 432 `E003_corrected_replay` unshuffled 512-residue Gold predictions, and 432 `E005_primary` broken-pair partner-shuffled 1024-residue control predictions across the six included test environments. `data/manifests/panel_manifest.json` records the historical score convention per cohort:

- `E003_corrected_replay` and `E005_core` clip probabilities to `[1e-7, 1 - 1e-7]` before AUROC and average precision. Clipping may create ties; scikit-learn's standard tie handling is preserved, with no score jitter or presort.
- `E005_primary` (the broken-pair partner-shuffled 1024-residue control) computes AUROC and average precision from the saved raw probabilities. Replacing this with clipping changes 8 of the 432 `E005_primary` cells and does not reproduce their stored metrics.

The cohorts therefore record their exact historical score convention in `data/manifests/panel_manifest.json`. The release recomputes each with its frozen convention and joins by unique `row_id` to its own panel labels.

| Metric | Input and rule | Availability |
|---|---|---|
| AUROC | Cohort-specific ranking convention above | Recomputed for all 1,296 cells in `results/pointwise/`, 432 cells in `results/shuffle_pointwise/`, 1,152 cells in `results/external_pointwise/`, and 864 cells in `results/common_512_pointwise/` |
| AUPRC | Same ranking convention; report with panel prevalence | Recomputed for all 1,296 cells in `results/pointwise/` and all 1,152 external cells |
| MCC / F1 / Balanced Accuracy | Thresholded score at original validation-selected threshold (`results/tables/validation_selected_thresholds_144_units.csv`) | Recomputed for all 864 unshuffled Gold cells (432 `E003_corrected_replay` + 432 `E005_core`) in `results/tables/reported_mcc.csv` |
| Brier/ECE | Probability clipped to `[0, 1]` | Recomputed from released score rows |
| NLL | Probability clipped to `[1e-7, 1 - 1e-7]` | Recomputed from released score rows |

No test-set threshold is selected. All 144 validation-selected thresholds (72 E003 units + 72 E005 units) and all 72 `E005_core` validation prediction columns (`results/validation_pointwise/E005_core/validation__{env}__all12_models.csv.gz`) are shipped. AUPRC is prevalence-sensitive; the positive fraction is recorded for every panel.

External predictions for all 8 surfaces (`vidal__measured`, `intact__measured`, `biogrid__uniform_partner`, `biogrid__tppni2025_scm_cl3_exact_balanced`, `biogrid__bernett2026_ilp_exact`, `string__uniform_partner`, `string__tppni2025_scm_cl3_exact_balanced`, `string__bernett2026_ilp_exact`) across both E003 (576 cells) and E005 (576 cells) are included in `results/external_pointwise/` plus all 24 Fully Random external prediction cells (`results/fully_random_external_pointwise/`, covering `vidal__measured` and `intact__measured`) are included and verified by `python scripts/verify_additional.py --scores`.

## Uncertainty

Bootstrap configuration manifests are shipped in `configs/bootstrap/e005_uncertainty_manifest.json` and `configs/bootstrap/e005_common_test_uncertainty_manifest.json` and verified by `python scripts/reproduce_bootstrap.py`. Paired row bootstrap resamples the same test rows for both score vectors (2,000 replicates, seed base 1729). The reported intervals measure test-row sensitivity. One-endpoint grouping is a separate sensitivity and does not model dependence through both endpoints. The three training seeds (11, 22, 33) describe those runs only and are not biological replicates.
