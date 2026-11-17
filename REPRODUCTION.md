# Reproduction scope

See `README.md` for copy-paste CPU commands and `docs/EVALUATION.md` for scoring conventions.

- `python scripts/verify_summary.py`: verifies core integrity, 56 metric-level claims, 11 release audit tables, figure source tables, and frozen summaries without requiring the large data archives.
- `python scripts/verify_sampler.py`: verifies deterministic sampler & schema contracts against `tests/fixtures/sampler_expected.json`.
- `python scripts/verify_final.py`: asserts all 22 current final-manuscript artwork hashes and 3 historical R2 figure hashes in `manifests/R2_FIGURES.csv`, all 44 panel manifests, 1,059 crosswalk rows, 114 checkpoint inventory entries, 72 Fully Random Gold + 24 Fully Random external score files, and the 6 exact Uniform/Uniform Main Figure 5 (pooled-ESM baseline) AUROCs.
- `python reproduce.py --all`: verifies all 1,296 pointwise Gold cells in `results/pointwise/`, all 864 validation-thresholded MCC cells (`432 E003_corrected_replay + 432 E005_core`), all 44 panel definitions/distributed files, 56 metric-level claims, 11 audit tables, 22 final artwork checks (plus 3 historical figure checks), supported figure redraws, and anonymity scan.
- `python scripts/verify_additional.py --scores`: verifies all 72 Fully Random Gold cells, all 24 Fully Random external cells (`96/96` Fully Random prediction-level verification), 432 unshuffled 1024 `E005_core` Gold cells, 432 corrected 512 shuffled cells, 1,152 external cells (`576 E003 + 576 E005`), 864 paired common-512 cells, and 72 `E005_core` validation prediction columns.
- `python scripts/reproduce_bootstrap.py --reproduce-matched-144 --workers 4`: recomputes all 144 matched 512-vs-1024 common-test paired bootstrap intervals (`2,000` stratified replicates, `seed = 1729 + test_index`).
- `python scripts/evaluate_predictions.py`: evaluates new prediction CSVs; defaults to `--cohort E005_core` (unshuffled 1024-residue Gold) and keeps `--cohort E005_primary` explicitly available for broken-pair partner-shuffling controls.

Across these commands, verification falls into three distinct tiers: (1) **recomputed from distributed predictions**, (2) **verified at metric level only** against frozen summary/audit tables, and (3) **not reproducible from this release** (training, GPU inference, excluded checkpoints/caches, historical E003 per-row validation scores, and raw third-party database dumps). No checkpoint binary, GPU, network download, or private path is required. Tested on Windows 11 (Python 3.11 CPU) and Linux (Python 3.11 CPU).
