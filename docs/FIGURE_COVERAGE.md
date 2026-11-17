# Manuscript figure reconciliation

`manifests/R2_FIGURES.csv` identifies all **22** exact PDF figures referenced in the final manuscript (5 main figures and 17 appendix figures), bound by `manifests/PAPER_ARTWORK_BINDING.json`, plus **3** superseded R2 PDFs retained in `figures/R2/` with `status=historical` for traceability (`25` PDFs total in `figures/R2/`, plus 11 convergence diagnostic PDFs in `figures/convergence/`). The manuscript source itself stays private. The archive's older figure-spec names are analysis-output names, not final manuscript figure numbers.

| Manuscript figure | Manuscript label | Included artifact and support |
|---|---|---|
| Main Figure 1 | `fig:design` | `figures/R2/figure_design.pdf` (schematic; historical layout code unavailable) |
| Main Figure 2 | `fig:factorial` | `figures/R2/fig02_original_submitted.pdf`; 144 source means in `results/additional/figure_data__fig02_factorial_delta_auprc.csv` |
| Main Figure 3 | `fig:pairdependence` | `figures/R2/fig03_corrected_shuffling_final.pdf`; supported redraw from `results/metric_level/corrected_shuffle_deltas.csv` |
| Main Figure 4 | `fig:measured-transfer` | `figures/R2/fig04_measured_512_only.pdf` (512-residue primary view); supported redraw from external metric tables |
| Main Figure 5 | `fig:baseline-comparison` | `figures/R2/fig06_pooled_esm_baselines.pdf` (pooled-ESM baseline comparison); exact source tables (`diagonal_comparison.csv`, `pooled_esm_summary.csv`) and `scripts/plot_figure6.py` |
| Appendix Figures 1–14 | `fig:supp-*` / `fig:construction-*` | 14 Appendix PDFs listed in `manifests/R2_FIGURES.csv` (`figS01`–`figS14`), metadata-sanitized with 0-pixel diff |
| Appendix Figure 15 | `fig:replication` | `figures/R2/figure_replication_final.pdf` (512-vs-1024 replication); matched metric rows (`MATCHED_ALL_144_INTERVALS.csv`) and `e005_retention.csv` |
| Appendix Figure 16 | `fig:supp-measured-full` | `figures/R2/fig04_absolute_measured_final.pdf` (full E003 + E005 measured transfer); `results/additional/EXTERNAL_ALL_SEED_METRICS.csv` |
| Appendix Figure 17 | `fig:supp-full-matrices` | `figures/R2/figure_full_matrices.pdf`; `PRIMARY_1024_ALL_GOLD_METRICS.csv`, `ORIGINAL_512_ALL_432_GOLD_SEED_ROWS.csv` |
| Historical (superseded R2) | `fig:pairdependence`, `fig:measured-transfer`, `fig:replication` | `fig03_corrected_shuffling.pdf`, `fig04_absolute_measured.pdf`, `figure_replication.pdf` (marked `status=historical` in `manifests/R2_FIGURES.csv`) |

Run `python reproduce.py --figures` and `python scripts/plot_figure6.py` for supported metric-based redraws. Outputs live in `results/figures_regenerated/` and do not overwrite frozen PDFs in `figures/R2/`. Figure validity does not imply raw training/checkpoint evidence exists for every panel.
