# Dataset versions, panels and acquisition

Read NOTICE.md for licenses, attribution and the date checked. All 44 panels in `data/manifests/panel_manifest_all44.json` have actual files in the data archive. The runtime manifest references those same files. Hashes describe distributed sanitized bytes, not private source paths or development revisions.

## Frozen objects

E003_corrected_replay has six environments for each train/validation/test split (18 panels). E005_core has the corresponding 18 panels at 1024 residues. E005_primary has six distinct broken-pair test panels. RBT_fully_random has two global-pair train/validation panels. This totals 44. Test environments retain row multiplicity; do not deduplicate or replace primary with core.

For E005_primary, score `broken_candidate_sequence_sha256` on the right, retaining the original row label. For the other panels, the right endpoint is `true_partner_sequence_sha256` for positive rows and `candidate_sequence_sha256` for negative rows. Use the released `panels.py` interface. Sequence SHA256 keys are biological content identifiers. Private source, registry and candidate-universe hashes are removed.

All Gold negative labels mean **synthetic unobserved pair**, not measured non-interaction. TPPNI and Bernett ILP here are implemented sampling methods on project inputs, not imported negative-dataset records. The unmodified Gold reference negatives do not constitute an anchor-fixed causal treatment. Assay non-detections in measured-source summaries remain distinct.

## Acquisition

`python scripts/acquire_gold.py --output <new-directory>` downloads seven pinned public Figshare v4 files and checks the provider MD5 and size. `data/manifests/gold_acquisition.json` records stable file IDs, URLs and provider checksums. This is an upstream acquisition helper, **not** a claim that reprocessing current downloads reproduces the historical materialized source CSVs. The exact project panels are supplied directly.

STRING source: v12.0 human physical links, obtainable through the version-12-0 STRING portal linked in the current version-history page. BioGRID source: 5.0.259 archive. IntAct source: human binary physical snapshot dated 2026-07-26, whose exact immutable public payload is not established. Do not substitute current downloads and call them exact historical reconstruction.

HuRI source: published HuRI/HI-union downloadable TSVs at https://interactome-atlas.org/download. The screen used a 52,068-pair frozen raw list; the current portal lists different counts. HGNC frozen mapping version is unavailable. Current data are useful for new analyses but do not reproduce the historical mapping.

Lambourne source: Molecular Cell 2025 Data S4 and clone annotations. The author-linked Zenodo deposit is pinned in `data/manifests/lambourne_acquisition.json` with its public checksum. It is approximately 3.77 GB and is not automatically downloaded. Its identity with the historical Data S4 transformation has not been established. Original external predictions are unavailable locally; downloading source assay data does not recreate them.

## Source-specific absent lists

No external IntAct, Vidal, BioGRID, STRING or HuRI interaction/pair lists, original HGNC database snapshot, or source exclusion-registry dumps are claimed distributed. Their absence is separate from the 44 **included** Gold-based panels. Exact source/mapping reconstruction remains incomplete. The provided metrics/audits are usable without those source dumps; their raw-input analyses cannot be rerun from aggregate tables alone.

The recovered author-generated ORF/gene/UniProt crosswalk is included as `data/mappings/vidal_orf_crosswalk.csv`. It preserves unresolved status and sequence-content keys. It is a mapping resource, not the missing external pair list or original HGNC database. Its crosswalk-only resolution path differs from the registry-first strata audit; do not infer stratum AUROC from this table. Attribution: Lambourne et al.; HGNC (CC0); UniProt (CC BY 4.0).
