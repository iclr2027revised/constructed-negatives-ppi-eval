# Anonymous Reproducibility Release

Code, frozen negative-construction panels, saved prediction tables, and verification scripts for **Constructed Negatives Shape the Evaluation of Sequence-Based Protein–Protein Interaction Models** (ICLR 2027 submission).

## What this benchmark investigates

Sequence-based protein–protein interaction (PPI) models are commonly trained and evaluated using synthetically constructed negative pairs alongside curated positives. This benchmark isolates how the choice of negative-construction method (Uniform, degree-matched, sequence-nuisance-matched, frozen-ESM-matched, global Bernett ILP, TPPNI, and two-endpoint Fully Random), sequence-length eligibility (`512` vs `1024` residues), and broken-pair partner shuffling affect within-cohort generalization, shortcut reliance, and external transfer to measured assays (Vidal/Y2H-iso, IntAct, BioGRID, and STRING).

## Repository and archive layout

Because the full compressed panel and per-pair prediction payload is ~2.37 GB, the release is split across the Git repository (core code, configs, summary tables, figures, and manifests — matching `anonymous_submission_core.zip`) and two <= 2 GiB data archives attached to the GitHub Release:

- **Core repository (`anonymous_submission_core.zip`, 206 files):**
  - `src/ppi_shift/`, `src/ppi_shift_release/`: negative-environment samplers, panel/prediction joiners, metric evaluation, and model interface definitions (`dscript_esm2`, `frozen_esm2_mlp`, `rapppid`).
  - `scripts/`, `reproduce.py`: CPU verification, integrity/anonymity checks, paired bootstrap (`reproduce_bootstrap.py`), and supported figure redraws.
  - `configs/`: historical YAML config snapshots (`e003_512.yaml`, `e005_1024.yaml`, `fully_random_1024.yaml`) and bootstrap replicate manifests (`configs/bootstrap/`).
  - `results/tables/`, `results/metric_level/`, `results/additional/`, `results/convergence/`: frozen summary tables, 864 thresholded MCC/F1/BA rows, 144 validation-selected thresholds, 144 matched 512-vs-1024 paired bootstrap intervals, 864 Vidal stratum rows, 96 Fully Random seed rows, and audit tables.
  - `figures/R2/`, `figures/convergence/`: all 21 metadata-sanitized manuscript PDFs and 11 convergence diagnostic PDFs (with `.png` and `.source.csv`).
  - `manifests/`, `docs/`, `tests/`: SHA-256 integrity manifests, 114-checkpoint inventory (`CHECKPOINT_INVENTORY.csv`), dataset/limitation/provenance docs, and 38 CPU tests.
- **Data archives (`anonymous_submission_data_part01.zip` + `anonymous_submission_data_part02.zip`, 1,471 files):**
  - `data/frozen_panels/`: all 44 frozen Gold-based train/validation/test `.csv.gz` pair lists (`E003_corrected_replay`, `E005_core`, `E005_primary`, `RBT_fully_random`).
  - `data/sequence_registry/sequences.fasta.gz`: 18,622 panel sequence headers (17,981 unique normalized sequences).
  - `results/pointwise/`: 1,296 per-pair Gold prediction files (432 `E005_core` unshuffled 1024, 432 `E003_corrected_replay` 512, 432 `E005_primary` partner-shuffled 1024).
  - `results/shuffle_pointwise/`: 432 `E003` corrected-512 partner-shuffled prediction columns.
  - `results/external_pointwise/`: 1,152 external-transfer prediction cells (576 `E003` + 576 `E005` across Vidal/Y2H, IntAct, BioGRID, STRING).
  - `results/common_512_pointwise/`: 864 paired 512-vs-1024 prediction cells on the identical frozen 512 test pairs.
  - `results/fully_random_pointwise/` & `results/fully_random_external_pointwise/`: 72 Gold + 24 external Fully Random prediction files (96/96 total).
  - `results/validation_pointwise/`: 72 `E005_core` validation prediction columns.

## Installation (CPU, Python 3.11)

Create a virtual environment and install the pinned CPU dependencies (`requirements-lock.txt`, plus `requirements-model-tests.txt` for PyTorch model-interface tests):

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install -r requirements-model-tests.txt
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
```

On Linux/macOS, activate with `source .venv/bin/activate` and set the same environment variables with `export`. Tested on Windows 11 (Python 3.11 CPU) and Linux (Python 3.11 CPU).

### Core-only verification (without data archives)

```bash
python scripts/verify_summary.py
python scripts/verify_sampler.py
python reproduce.py --figures
```

## Downloading and merging the data archives

To run full prediction-level verification, download `anonymous_submission_data_part01.zip` and `anonymous_submission_data_part02.zip` (and their `.sha256` sidecars) from the GitHub Release, verify checksums, and extract both archives into the parent directory of your cloned repository (or extract `anonymous_submission_core.zip`, `anonymous_submission_data_part01.zip`, and `anonymous_submission_data_part02.zip` into the same parent directory; all three share `anonymous_submission_repo/` as their root). See `manifests/ASSET_FILES.json` for payload hashes.

## Full reproduction, verification, and tests

After merging the core repository and data archives:

```bash
python reproduce.py --verify
python reproduce.py --smoke
python reproduce.py --all
python -m pytest tests/ -q -p no:cacheprovider
python scripts/verify_final.py
python scripts/verify_additional.py --scores
python scripts/reproduce_bootstrap.py --reproduce-matched-144 --workers 4
python scripts/plot_figure6.py
```

What each check verifies:
- `reproduce.py --verify`: SHA-256 integrity of all 1,676 tracked files (`manifests/INTEGRITY.json`), all 44 panel files, sequence-registry hashes, and generic anonymity patterns.
- `reproduce.py --smoke`: fast CPU subset verification (sampled panels, pointwise cells, MCC cells, and 56 metric-level claims).
- `reproduce.py --all`: full CPU recomputation of all 1,296 Gold pointwise cells, 864 validation-thresholded MCC cells, 56 frozen-table claims, 11 audit tables, 21 manuscript artwork hashes, supported figure redraws (`results/figures_regenerated/`), and anonymity checks.
- `pytest tests/ -q -p no:cacheprovider`: 38 unit and package-contract tests (integrity, panel/sampler determinism, metric conventions, model interfaces, and evaluation).
- `scripts/verify_additional.py --scores`: prediction-level verification of all 96/96 Fully Random files (72 Gold + 24 external), 432 `E005_core` Gold cells, 432 `E003` corrected-512 shuffled cells, 1,152 external cells, 864 paired common-512 cells, and 72 `E005_core` validation columns.
- `scripts/reproduce_bootstrap.py --reproduce-matched-144 --workers 4`: deterministic 2,000-replicate paired bootstrap recomputation of all 144 matched 512-vs-1024 intervals in `results/additional/MATCHED_ALL_144_INTERVALS.csv`.

## Reproducibility scope and limitations

- **Recomputed from distributed predictions:** All 1,296 Gold cells (`E003_corrected_replay`, `E005_core`, `E005_primary`), 432 `E003` corrected-512 shuffled cells, 1,152 external cells, 864 paired common-512 cells, 96/96 Fully Random cells, 72 `E005_core` validation columns, 864 thresholded MCC/F1/BA cells, 432 Vidal exact-sequence stratum rows, and 144 matched paired bootstrap intervals.
- **Verified at metric level only:** Symmetric endpoint-control deltas (`shortcut_delta_symmetric.csv`), control-vs-neural sign agreement, DataSAIL leakage-reduced split sensitivity (`e005_datasail_metrics.csv`), collision/HuRI/homology aggregate audit tables, and original 512 historical table baselines (`ORIGINAL_512_ALL_432_GOLD_SEED_ROWS.csv`), because those control fits or raw third-party dumps are not redistributed as runnable inference pipelines.
- **Not reproducible from this release and why:**
  - **Model training and GPU inference:** All 114 trained checkpoint binaries (`.pt`, inventoried in `manifests/CHECKPOINT_INVENTORY.csv`), pretrained ESM-2 weights/token caches, and exact historical D-SCRIPT/TUnA training adapters are excluded.
  - **Historical `E003` per-row validation predictions (72 units):** Never saved on the cluster; scalar validation-selected thresholds (`validation_selected_thresholds_144_units.csv`) and test-panel MCC/F1/BA (`reported_mcc.csv`) are supplied and verified instead.
  - **Raw third-party database dumps & unsplit cluster files:** Raw BioGRID, STRING, IntAct, HuRI, HGNC, and Lambourne Data S4 archives, the full 50,398-protein `sequence_registry.csv`, and an unsplit ~1.6 GB cluster shuffle text export are excluded (covered by the supplied crosswalk, panel-specific sequence registry, and split `.csv.gz` prediction files).

See `LIMITATIONS.md`, `REPRODUCTION.md`, `docs/DATASETS.md`, `docs/PROCEDURES.md`, and `manifests/RESOURCE_INVENTORY.csv` for full schema and cohort details.

## Licensing and third-party attribution

- **Code** (`src/`, `scripts/`, `tests/`, `configs/`, `reproduce.py`): **MIT License** (see `LICENSE`).
- **Original documentation, computed metrics, saved prediction outputs (`results/**`), and figures**: **CC BY 4.0**.
- **Third-party datasets and upstream methods** (Bernett et al. Gold benchmark, UniProt, UPNA-PPI/TPPNI, IntAct/IMEx, STRING, BioGRID, HuRI, HGNC, Lambourne et al. 2025 Data S4): retained with full attribution and primary license terms in `NOTICE.md` and `docs/ATTRIBUTION.md`.
