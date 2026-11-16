# Executed model definitions

This document records the four architecture identities behind the reported results. It does not claim upstream parity where the executed implementation differed.

| Model | Adapter source in package | Weights in package | Executed identity |
|---|---|---|---|
| Frozen-ESM2 pair MLP | Yes (`frozen_esm2_mlp.py`) | No | ESM2-150M 640-dimensional frozen means; symmetric `[a+b, abs(a-b), a*b]` features; 64/32 MLP |
| RAPPPID | Yes (`rapppid.py`) | No | AWD-LSTM twin encoder; 64 recurrent units per direction; order-sensitive concatenation head |
| D-SCRIPT adapter | Yes (`dscript_esm2.py`) | No | ESM2-150M (`facebook/esm2_t30_150M_UR50D`, layer 30, 640-d) per-residue tokens projected to 100-d (`projection.0.weight` shape `(100, 640)` = 256,000 bytes), contact CNN |
| TUnA adapter | No | No | ESM2-150M (`facebook/esm2_t30_150M_UR50D`, layer 30, 640-d) Transformer/SNGP adapter; no StepLR or GP mean-field correction in the executed path |

## D-SCRIPT ESM2-150M vs ESM2-650M provenance resolution

Every one of the 42 trained `dscript_esm2` checkpoints across all experiment cohorts (9 in E001, 3 in E002, 12 in E003 512-residue, and 18 in E005 1024-residue, including the 6 Fully Random units) used **ESM2-150M** (`facebook/esm2_t30_150M_UR50D`, layer 30, 640-dimensional per-residue representations), **not** `ESM2-650M` (`esm2_t33_650M_UR50D`, 1280-dimensional):

1. **Run configs and unit provenance**: `token_encoder: esm2_t30_150M_UR50D_layer30` in `configs/e005_core4_1024.yaml` and all 42 unit provenance manifests (`results/tables/dscript_esm2_backbone_provenance_42_units.csv`).
2. **Token cache contracts**: `embedding_dim = 640`, `layer = 30`, `model_name = "facebook/esm2_t30_150M_UR50D"`.
3. **Checkpoint tensor dimensions**: in all 42 `.pt` checkpoint archives, `projection.0.weight` (`archive/data/4`) is 256,000 bytes = 64,000 `float32` elements = shape `(100, 640)`. An ESM2-650M input would require shape `(100, 1280)` = 512,000 bytes.

## What can be inspected

The included model files document the released architectures and their feature contracts. They do not contain trained weights. New prediction files can be evaluated against the distributed panels using the package evaluation interface.