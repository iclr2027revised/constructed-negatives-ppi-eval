# Aggregate audit tables

This private-review candidate includes 11 CSVs under `results/metric_level/`. They preserve already computed aggregate evidence. They do not include upstream interaction databases, withheld pair lists, or per-pair scores for these audit analyses. `reproduce.py --verify`, `--smoke`, and `--all` check their file presence, frozen row counts, and the listed arithmetic. Those checks do not turn a metric-level table into a prediction-level reproduction.

## Collision and mapping evidence

- `known_positive_collision_audit.csv` has 96 rows: 24 each for BioGRID, STRING, IntAct, and HuRI. The BioGRID, STRING, and IntAct rows retain the `MEASURED_POSTHOC` status and their measured collision counts. The HuRI rows remain `SOURCE_UNAVAILABLE` with empty collision fields. They must not be read as measured zeros.
- `huri_collision_audit.csv` preserves the separate corrected HuRI screen across the frozen cohort, split, and sampler panels. Its 560,382 negative panel rows partition into 185,737 comparable pairs, 304 pairs with ambiguous endpoints, and 374,341 pairs outside the screenable universe. The audit reports zero confirmed positive collisions among the 185,737 comparable pairs. This is not a biological false-negative rate, and the unscreenable rows remain untested.
- `huri_mapping_coverage.csv` records mapping counts: 8,080 resolved, 7 ambiguous, and 158 unmapped genes; 50,489 resolved, 34 ambiguous, and 1,545 unmapped source pairs; and the benchmark-negative partition above.
- `huri_collisions_and_controls.csv` retains the 24 aggregate controls rows and their per-row denominator partition. It includes positive controls as a check on source mapping; the control recall is not an estimate of negative-pair sensitivity.

These author-computed aggregate tables are distributed under CC BY 4.0; source attribution is in NOTICE.md.

## Homology summaries

- `homology_sample_alignment_summary.csv` records a frozen 2,000-row negative sample for each cohort and sampler. For ESM-matched negatives, 1,988 E003 rows and 1,986 E005 rows aligned; 166 and 164, respectively, met the filter of at least 30% identity and at least 80% coverage of both sequences.
- `homology_sample_scores.csv` records the existing filtered metric summaries. Each E003 and E005 comparison retains 1,822 balanced pairs at prevalence 0.5. These values are sourced from saved summaries; underlying aligned pair rows and score vectors are not included.
- `homology_full_panel_scores.csv` is a separate full-panel analysis: E003 has 10,051 measured rows, 9,204 retained, 60 unavailable, and 847 removed as close; E005 has 20,663 measured, 18,741 retained, 155 unavailable, and 1,922 removed. Its 500 bootstrap replicates group on one anchor endpoint. This does not account fully for dependence through both endpoints or supply independent biological confirmation.

The sampled and full-panel analyses are distinct. Neither shows that sequence-identity novelty means a new protein family or strict out-of-distribution generalization.

## Endpoint distribution and baseline summaries

- `endpoint_pair_concentration.csv` records frozen panel-level duplicate and reuse counts. On E005 tests, the ESM-matched negatives have 996 repeated unordered negative-pair rows and maximum candidate-endpoint reuse 172; Uniform has 129 and 22.
- `endpoint_frequency_audit.csv` summarizes sequence and endpoint features for E005 train and validation rows. In training negatives, the maximum endpoint degree is 401 for Uniform and 51 for Fully Random. This descriptive contrast is not evidence of a causal degree mechanism.
- `negative_distribution_contrast.csv` records Fully Random minus Uniform feature differences. For negative rows, `delta_degree_max` is -350 in training and -215 in validation. Training and validation both change in the Fully Random arm.
- `additive_baselines.csv` preserves the existing AUROC/AUPRC summaries for four model baselines and the strictly additive pooled-ESM control, including each row's reported run count and SD where defined. These summaries do not include prediction rows or validation-selected thresholds.

## Rights and provenance boundary

These author-computed aggregate tables are distributed under CC BY 4.0; source attribution is in NOTICE.md.
