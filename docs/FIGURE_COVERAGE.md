# R2 figure reconciliation

`manifests/R2_FIGURES.csv` identifies all **21** exact R2 PDF figures referenced in the PI-approved revised manuscript (`main_R2.tex`) and their released-byte hashes. All 21 files live in `figures/R2/` (plus 11 convergence diagnostic PDFs in `figures/convergence/`); the private manuscript itself stays in the private author handoff. The archive's older figure-spec names are analysis-output names, not R2 figure numbers.

| R2 figure | Manuscript label | Included artifact and support |
|---|---|---|
| Main Figure 1 | `fig:design` | `figures/R2/figure_design.pdf` (schematic; historical layout code unavailable) |
| Main Figure 2 | `fig:factorial` | `figures/R2/fig02_original_submitted.pdf`; 144 source means in `results/additional/figure_data__fig02_factorial_delta_auprc.csv` |
| Main Figure 3 | `fig:pairdependence` | `figures/R2/fig03_corrected_shuffling.pdf`; supported redraw from `results/metric_level/corrected_shuffle_deltas.csv` |
| Main Figure 4 | `fig:measured-transfer` | `figures/R2/fig04_absolute_measured.pdf`; supported redraw from external metric tables |
| Main Figure 5 | `fig:replication` | `figures/R2/figure_replication.pdf`; matched metric rows (`MATCHED_ALL_144_INTERVALS.csv`) and `e005_retention.csv` |
| Main Figure 6 | `fig:baseline-comparison` | `figures/R2/fig06_pooled_esm_baselines.pdf`; exact source tables (`diagonal_comparison.csv`, `pooled_esm_summary.csv`) and `scripts/plot_figure6.py` |
| Appendix Figures 1–15 | `fig:supp-*` / `fig:construction-*` | All 15 Appendix PDFs listed in `manifests/R2_FIGURES.csv` (`figS01`–`figS14` + `figure_full_matrices.pdf`), metadata-sanitized with 0-pixel diff |

Run `python reproduce.py --figures` and `python scripts/plot_figure6.py` for supported metric-based redraws. Outputs live in `results/figures_regenerated/` and do not overwrite frozen R2 PDFs. Figure validity does not imply raw training/checkpoint evidence exists for every panel.
