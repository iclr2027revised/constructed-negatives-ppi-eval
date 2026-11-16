# Verification modes

Core only: `python scripts/verify_summary.py` verifies the core manifest, 56 metric-level claims, 11 aggregate audit tables, supported figure-source arithmetic, all Fully Random comparison rows and Vidal summary denominators. `python reproduce.py --figures` redraws supported metric figures.

Core plus data: `python reproduce.py --verify` verifies integrity, all 44 panels and sequence joins. `python reproduce.py --all` additionally recomputes all 1,296 main Gold score cells (432 `E003_corrected_replay`, 432 `E005_core`, and 432 `E005_primary`) and 864 validation-thresholded MCC cells, validates summary claims and redraws supported figures. `python scripts/verify_additional.py --scores` recomputes 72 Fully Random Gold + 24 Fully Random external AUROC/AUPRC cells using E005_core plus all recovered shuffle/external/common-512/validation prediction files. None of these commands runs a model fit or downloads data.

The integrity files contain hashes of distributed bytes only. The source acquisition manifest contains public provider checksums. Independent private validation additionally scans decompressed payloads and PDF metadata; logs are outside the repository. Full mode intentionally fails if the data archive has not been installed. Summary mode does not label absent scores verified.

Use `PYTHONDONTWRITEBYTECODE=1` and pytest `-p no:cacheprovider` to avoid caches embedding machine paths. Generated figure outputs and validation reports are excluded from immutable input inventories.
