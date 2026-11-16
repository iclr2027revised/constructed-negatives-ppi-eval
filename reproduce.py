#!/usr/bin/env python
"""Single entrypoint for reproducing the sequence-benchmark results.

    python reproduce.py --smoke     # ~1 minute, CPU only, PASS/FAIL
    python reproduce.py --all       # full results reproduction, CPU only
    python reproduce.py --figures   # regenerate the released figures
    python reproduce.py --verify    # integrity and anonymity check only

Every mode reads only files inside this repository.  No GPU, no cluster, no
private checkout and no network access is required.

What each mode proves is stated in ``docs/REPRODUCTION.md``.  The short version:
``--all`` recomputes every reported metric from the released predictions and
metric tables and compares each one with the value recorded in
``manifests/EXPECTED_VALUES.json``.  It does not re-run any model, because the
checkpoints are not redistributed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

REQUIRED_PYTHON = (3, 11)
REQUIRED_MODULES = ("numpy", "pandas", "sklearn", "matplotlib")
VALIDATION_THRESHOLD_SOURCES = frozenset(
    {"matched_validation_max_mcc", "reused_e001_validation_max_mcc"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_environment() -> dict[str, Any]:
    """Confirm the interpreter and the four runtime dependencies."""

    problems: list[str] = []
    if sys.version_info[:2] < REQUIRED_PYTHON:
        problems.append(
            f"Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ required, "
            f"found {platform.python_version()}"
        )
    versions: dict[str, str] = {}
    for name in REQUIRED_MODULES:
        try:
            module = __import__(name)
        except ImportError:
            problems.append(f"missing dependency: {name}")
            continue
        versions[name] = getattr(module, "__version__", "unknown")
    return {
        "name": "environment",
        "status": "PASS" if not problems else "FAIL",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": versions,
        "problems": problems,
    }


def check_integrity(*, limit: int | None = None) -> dict[str, Any]:
    """Verify every file against ``manifests/INTEGRITY.json``."""

    manifest_path = ROOT / "manifests/INTEGRITY.json"
    if not manifest_path.exists():
        return {
            "name": "integrity",
            "status": "FAIL",
            "problems": ["manifests/INTEGRITY.json is missing"],
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["files"]
    problems: list[str] = []
    checked = 0
    for relative, entry in sorted(entries.items()):
        if limit is not None and checked >= limit:
            break
        path = ROOT / relative
        if not path.exists():
            problems.append(f"missing file: {relative}")
            continue
        if path.stat().st_size != entry["bytes"]:
            problems.append(f"size mismatch: {relative}")
            continue
        if _sha256(path) != entry["sha256"]:
            problems.append(f"hash mismatch: {relative}")
            continue
        checked += 1
    return {
        "name": "integrity",
        "status": "PASS" if not problems else "FAIL",
        "files_listed": len(entries),
        "files_checked": checked,
        "problems": problems[:20],
    }


def check_panels(*, limit: int | None = None) -> dict[str, Any]:
    """Verify the frozen panels and the sequence registry."""

    from ppi_shift_release import panels

    entries = panels.load_manifest(ROOT)
    if limit is not None:
        entries = entries[:limit]
    problems: list[str] = []
    verified = 0
    for entry in entries:
        result = panels.verify_panel(ROOT, entry)
        if result.get("status") != "OK":
            problems.append(f"{entry.cohort}/{entry.environment}: {result.get('status')}")
            continue
        verified += 1
    registry = panels.verify_sequence_registry(ROOT)
    if registry["mismatched"]:
        problems.append(
            f"sequence registry: {registry['mismatched']} entries do not hash to their key"
        )
    return {
        "name": "panels",
        "status": "PASS" if not problems else "FAIL",
        "panels_verified": verified,
        "panels_listed": len(entries),
        "sequence_records": registry["sequences"],
        "problems": problems[:20],
    }


def check_full_panel_manifest() -> dict[str, Any]:
    """Check all 44 split records and every pair list shipped in this release."""

    path = ROOT / "data/manifests/panel_manifest_all44.json"
    problems: list[str] = []
    if not path.exists():
        return {
            "name": "full_panel_manifest",
            "status": "FAIL",
            "panels_listed": 0,
            "pairs_distributed": 0,
            "hash_only": 0,
            "problems": ["data/manifests/panel_manifest_all44.json is missing"],
        }

    try:
        entries = json.loads(path.read_text(encoding="utf-8"))["panels"]
    except (json.JSONDecodeError, KeyError) as error:
        return {
            "name": "full_panel_manifest",
            "status": "FAIL",
            "panels_listed": 0,
            "pairs_distributed": 0,
            "hash_only": 0,
            "problems": [f"invalid full-panel manifest: {error}"],
        }

    identities = [
        (entry.get("cohort"), entry.get("split"), entry.get("environment"))
        for entry in entries
    ]
    if len(entries) != 44:
        problems.append(f"expected 44 panel records, found {len(entries)}")
    if len(identities) != len(set(identities)):
        problems.append("duplicate cohort/split/environment record")

    distributed = 0
    hash_only = 0
    blocked_rights = 0
    missing = 0
    conditional_rights = 0
    source_recoverable = 0
    for entry in entries:
        relative = entry.get("path", "")
        panel_path = ROOT / relative
        availability = entry.get("availability")
        expected_convention = (
            "probability_clip_1e-7"
            if entry.get("cohort") in {"E003_corrected_replay", "E005_core"}
            else "raw_probability"
        )
        if entry.get("score_convention") != expected_convention:
            problems.append(f"score convention mismatch: {relative}")
        if entry.get("pair_duplicate_policy") != (
            "Preserve frozen row multiplicity; do not deduplicate; retain labels as frozen."
        ):
            problems.append(f"pair duplicate policy is missing or changed: {relative}")
        if entry.get("pair_identity_basis") != (
            "unordered pair of scored endpoint sequence SHA256 values"
        ):
            problems.append(f"pair identity basis is missing or changed: {relative}")
        expected_label_policy = (
            "Original row label remains attached after endpoint shift; mixed labels for a scored pair are retained."
            if entry.get("cohort") == "E005_primary"
            else "Mixed labels for the same scored sequence pair fail source validation."
        )
        if entry.get("pair_label_policy") != expected_label_policy:
            problems.append(f"pair label policy is missing or changed: {relative}")
        for field in (
            "unordered_pair_duplicate_groups",
            "unordered_pair_duplicate_extra_rows",
            "unordered_pair_mixed_label_groups",
            "scored_self_pair_rows",
        ):
            value = entry.get(field)
            if not isinstance(value, int) or value < 0:
                problems.append(f"invalid {field}: {relative}")
        if entry.get("cohort") != "E005_primary" and entry.get(
            "unordered_pair_mixed_label_groups"
        ) != 0:
            problems.append(f"unexpected mixed-label scored pairs: {relative}")
        if availability == "PAIRS_DISTRIBUTED":
            distributed += 1
            if not panel_path.is_file():
                problems.append(f"distributed panel is missing: {relative}")
            elif _sha256(panel_path) != entry.get("sha256"):
                problems.append(f"distributed panel hash mismatch: {relative}")
        elif availability == "HASH_ONLY":
            hash_only += 1
            if entry.get("sha256") is not None:
                problems.append(f"private source hash exposed for hash-only panel: {relative}")
            if panel_path.exists():
                problems.append(f"hash-only panel unexpectedly has a pair file: {relative}")
        elif availability == "SOURCE_RECOVERABLE_BUT_NOT_CLEARED":
            source_recoverable += 1
            if entry.get("sha256") is not None or panel_path.exists():
                problems.append(f"uncleared pair list has distributable bytes: {relative}")
            if entry.get("rights_status") not in {"UNKNOWN", "BLOCKED"}:
                problems.append(f"uncleared pair list has an inconsistent rights status: {relative}")
            blocked_rights += 1
        elif availability == "BLOCKED_RIGHTS":
            blocked_rights += 1
            if entry.get("sha256") is not None or panel_path.exists():
                problems.append(f"rights-blocked panel has distributable bytes: {relative}")
        elif availability == "MISSING":
            missing += 1
            if entry.get("sha256") is not None or panel_path.exists():
                problems.append(f"missing panel has unexpected bytes: {relative}")
        else:
            problems.append(f"unknown panel availability {availability!r}: {relative}")
        conditional_rights += entry.get("rights_status") == "CONDITIONAL"

    runtime_path = ROOT / "data/manifests/panel_manifest.json"
    runtime_entries = []
    try:
        runtime_entries = json.loads(runtime_path.read_text(encoding="utf-8"))["panels"]
    except (OSError, json.JSONDecodeError, KeyError) as error:
        problems.append(f"invalid runtime panel manifest: {error}")
    all_paths = {
        entry.get("path")
        for entry in entries
        if entry.get("availability") == "PAIRS_DISTRIBUTED"
    }
    runtime_paths = {entry.get("path") for entry in runtime_entries}
    if runtime_paths != all_paths:
        problems.append("runtime manifest does not cover exactly the distributed panel files")
    if len(entries) != 44:
        problems.append(f"expected 44 panel records, found {len(entries)}")
    return {
        "name": "full_panel_manifest",
        "status": "PASS" if not problems else "FAIL",
        "panels_listed": len(entries),
        "pairs_distributed": distributed,
        "hash_only": hash_only,
        "blocked_rights": blocked_rights,
        "missing": missing,
        "conditional_rights": conditional_rights,
        "source_recoverable_but_not_cleared": source_recoverable,
        "problems": problems[:20],
    }


def check_pointwise(*, limit: int | None = None, tolerance: float = 1e-9) -> dict[str, Any]:
    """Recompute reported metrics from the released pointwise predictions."""

    from ppi_shift_release import panels, predictions

    entries = panels.load_manifest(ROOT)
    index = {
        (entry.cohort, entry.environment): entry
        for entry in entries
        if entry.split == "test"
    }

    reported_path = ROOT / "results/tables/reported_metrics.csv"
    reported: dict[tuple[str, ...], dict[str, str]] = {}
    if reported_path.exists():
        import csv

        with reported_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row["cohort"],
                    row["model"],
                    row["train_environment"],
                    row["seed"],
                    row["test_environment"],
                )
                reported[key] = row

    base = ROOT / "results/pointwise"
    cells = sorted(base.rglob("*.csv.gz"))
    if limit is not None:
        cells = cells[:limit]

    problems: list[str] = []
    matched = 0
    actual_keys: list[tuple[str, ...]] = []
    for path in cells:
        parts = path.relative_to(base).parts
        if len(parts) != 5:
            problems.append(f"unexpected prediction path layout: {path.relative_to(base)}")
            continue
        cohort, model, train_environment, seed_dir, filename = parts
        seed = seed_dir.replace("seed-", "")
        test_environment = filename.split(".")[0]
        key = (cohort, model, train_environment, seed, test_environment)
        actual_keys.append(key)
        entry = index.get((cohort, test_environment))
        if entry is None:
            problems.append(f"{cohort}/{test_environment}: no frozen panel")
            continue
        try:
            table = predictions.load_predictions(path)
            summary = predictions.evaluate(ROOT, entry, table, bootstrap_replicates=0)
        except (OSError, ValueError, KeyError) as error:
            problems.append(f"{path.relative_to(ROOT)}: invalid prediction rows: {error}")
            continue
        expected = reported.get(key)
        if expected is None:
            problems.append(
                f"{cohort}/{model}/{train_environment}/{seed}/{test_environment}"
                ": no reported row"
            )
            continue
        auroc_diff = abs(summary["auroc"] - float(expected["auroc"]))
        auprc_diff = abs(summary["auprc"] - float(expected["auprc"]))
        if max(auroc_diff, auprc_diff) > tolerance:
            problems.append(
                f"{cohort}/{model}/{train_environment}/{seed}/{test_environment}: "
                f"AUROC {summary['auroc']:.12f} vs {expected['auroc']}, "
                f"AUPRC {summary['auprc']:.12f} vs {expected['auprc']}"
            )
            continue
        matched += 1
    if len(actual_keys) != len(set(actual_keys)):
        problems.append("duplicate prediction cell identity")
    if limit is None:
        if len(cells) != len(reported):
            problems.append(
                f"prediction/report cell count differs: {len(cells)} files, {len(reported)} reported rows"
            )
        missing = sorted(set(reported) - set(actual_keys))
        if missing:
            problems.append(f"{len(missing)} reported prediction cells have no score file")
        unreported = sorted(set(actual_keys) - set(reported))
        if unreported:
            problems.append(f"{len(unreported)} score files have no reported metric row")
    return {
        "name": "pointwise",
        "status": "PASS" if not problems else "FAIL",
        "cells_checked": len(cells),
        "cells_matched": matched,
        "tolerance": tolerance,
        "problems": problems[:20],
    }


def check_mcc(*, limit: int | None = None, tolerance: float = 1e-9) -> dict[str, Any]:
    """Recompute only MCC cells with released validation-owned thresholds."""

    from ppi_shift_release import metrics, panels, predictions

    path = ROOT / "results/tables/reported_mcc.csv"
    if not path.is_file():
        return {"name": "mcc", "status": "FAIL", "problems": ["reported_mcc.csv is missing"]}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    unique_rows: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = (
            row["cohort"],
            row["test_cohort"],
            row["model"],
            row["train_environment"],
            row["seed"],
            row["test_environment"],
        )
        if key in unique_rows:
            return {"name": "mcc", "status": "FAIL", "problems": [f"duplicate MCC cell: {key}"]}
        unique_rows[key] = row
    items = list(unique_rows.items())
    if limit is not None:
        items = items[:limit]

    entries = {
        (entry.cohort, entry.environment): entry
        for entry in panels.load_manifest(ROOT)
        if entry.split == "test"
    }
    problems: list[str] = []
    matched = 0
    for key, row in items:
        cohort, test_cohort, model, train_environment, seed, test_environment = key
        entry = entries.get((test_cohort, test_environment))
        if entry is None:
            problems.append(f"no frozen test panel for MCC cell: {key}")
            continue
        if row.get("threshold_source") not in VALIDATION_THRESHOLD_SOURCES:
            problems.append(f"MCC threshold is not validation-owned: {key}")
            continue
        if row.get("score_convention") != entry.score_convention:
            problems.append(f"MCC score convention differs from its panel: {key}")
            continue
        prediction_path = (
            ROOT
            / "results/pointwise"
            / cohort
            / model
            / train_environment
            / f"seed-{seed}"
            / f"{test_environment}.csv.gz"
        )
        try:
            table = predictions.load_predictions(prediction_path)
            joined = predictions.join_to_panel(ROOT, entry, table)
            observed = metrics.mcc_at_threshold(
                joined.labels,
                joined.scores,
                threshold=float(row["threshold"]),
                score_convention=entry.score_convention,
            )
        except (OSError, ValueError, KeyError) as error:
            problems.append(f"{prediction_path.relative_to(ROOT)}: invalid MCC input: {error}")
            continue
        if abs(observed - float(row["mcc"])) > tolerance:
            problems.append(f"MCC mismatch for {key}: {observed} vs {row['mcc']}")
            continue
        matched += 1
    if limit is None and len(rows) != 864:
        problems.append(f"expected 864 validation-threshold MCC rows, found {len(rows)}")
    return {
        "name": "mcc",
        "status": "PASS" if not problems else "FAIL",
        "mcc_cells_checked": len(items),
        "mcc_cells_matched": matched,
        "tolerance": tolerance,
        "problems": problems[:20],
    }


def check_figure_sources(*, tolerance: float = 1e-3) -> dict[str, Any]:
    """Check the metric tables used to redraw corrected shuffle and pooled figures."""

    problems: list[str] = []
    metric_root = ROOT / "results/metric_level"
    try:
        with (metric_root / "corrected_shuffle_deltas.csv").open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            shuffle = list(csv.DictReader(handle))
        with (metric_root / "pooled_esm_matched.csv").open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            pooled = list(csv.DictReader(handle))
    except OSError as error:
        return {"name": "figure_sources", "status": "FAIL", "problems": [str(error)]}

    shuffle_keys = [
        (
            row["cohort_label"],
            row["model_id"],
            row["seed"],
            row["train_environment"],
            row["test_environment"],
        )
        for row in shuffle
    ]
    if len(shuffle) != 864 or len(shuffle_keys) != len(set(shuffle_keys)):
        problems.append(f"corrected shuffle table must contain 864 unique cells, found {len(shuffle)}")
    for cohort, expected_rows in (("E003", 432), ("E005", 432)):
        rows = [row for row in shuffle if row["cohort_label"] == cohort]
        if len(rows) != expected_rows:
            problems.append(f"{cohort} corrected shuffle table has {len(rows)} rows, expected {expected_rows}")
        for row in rows:
            try:
                difference = float(row["auroc_original"]) - float(row["auroc_shuffled"])
                if abs(difference - float(row["delta_auroc"])) > 1e-12:
                    problems.append(f"shuffle delta arithmetic mismatch in {cohort}")
                    break
                # The frozen E005 metric source leaves ``rows`` blank. Keep
                # that source omission visible; do not backfill it from another
                # manifest or imply pair-level verification for this table.
                if cohort == "E003" and int(float(row["rows"])) != 20222:
                    problems.append(f"shuffle row count mismatch in {cohort}")
                    break
                if cohort == "E005" and row["rows"].strip():
                    problems.append("E005 shuffle source unexpectedly contains a row count")
                    break
            except (KeyError, ValueError) as error:
                problems.append(f"invalid shuffle source row: {error}")
                break

    pooled_keys = [
        (row["model"], row["environment"], row["seed"], row["metric"])
        for row in pooled
    ]
    if len(pooled) != 180 or len(pooled_keys) != len(set(pooled_keys)):
        problems.append(f"pooled ESM table must contain 180 unique cells, found {len(pooled)}")
    for row in pooled:
        try:
            value = float(row["value"])
            if not 0.0 <= value <= 1.0:
                problems.append("pooled ESM metric falls outside [0, 1]")
                break
        except (KeyError, ValueError) as error:
            problems.append(f"invalid pooled ESM source row: {error}")
            break
    expected_pooled_models = {
        "pooled_ESM_logistic_sum_absdiff",
        "symmetric_frozen_esm2_pair_mlp",
        "tuna_corrected",
        "dscript_esm2",
        "rapppid",
    }
    expected_environments = {
        "uniform_partner",
        "degree_partner_matched",
        "sequence_nuisance_matched",
        "esm_partner_matched",
        "bernett2026_ilp_exact",
        "tppni2025_scm_cl3_exact_balanced",
    }
    expected_pooled_keys = {
        (model, environment, seed, metric)
        for model in expected_pooled_models
        for environment in expected_environments
        for seed in ("11", "22", "33")
        for metric in ("auroc", "auprc")
    }
    if set(pooled_keys) != expected_pooled_keys:
        problems.append("pooled ESM table does not cover every model/environment/seed/metric cell")

    shuffle_claims = (
        ("E003", "symmetric_frozen_esm2_pair_mlp", "uniform_partner", 0.159),
        ("E003", "tuna_corrected", "uniform_partner", 0.207),
        ("E005", "symmetric_frozen_esm2_pair_mlp", "uniform_partner", 0.134),
        ("E005", "tuna_corrected", "uniform_partner", 0.182),
        ("E005", "symmetric_frozen_esm2_pair_mlp", "esm_partner_matched", 0.020),
        ("E005", "tuna_corrected", "esm_partner_matched", 0.026),
    )
    claim_matches = 0
    for cohort, model, test_environment, expected in shuffle_claims:
        values = [
            float(row["delta_auroc"])
            for row in shuffle
            if row["cohort_label"] == cohort
            and row["model_id"] == model
            and row["train_environment"] == "uniform_partner"
            and row["test_environment"] == test_environment
        ]
        observed = sum(values) / len(values) if values else float("nan")
        if len(values) != 3 or not math.isfinite(observed) or abs(observed - expected) > tolerance:
            problems.append(f"manuscript shuffle claim mismatch: {cohort}/{model}/{test_environment}")
        else:
            claim_matches += 1

    pooled_claims = (
        ("uniform_partner", "pooled_ESM_logistic_sum_absdiff", "auroc", 0.643),
        ("uniform_partner", "symmetric_frozen_esm2_pair_mlp", "auroc", 0.691),
        ("uniform_partner", "tuna_corrected", "auroc", 0.723),
        ("uniform_partner", "dscript_esm2", "auroc", 0.581),
        ("uniform_partner", "rapppid", "auroc", 0.525),
        ("tppni2025_scm_cl3_exact_balanced", "pooled_ESM_logistic_sum_absdiff", "auroc", 0.720),
        ("tppni2025_scm_cl3_exact_balanced", "symmetric_frozen_esm2_pair_mlp", "auroc", 0.784),
        ("tppni2025_scm_cl3_exact_balanced", "tuna_corrected", "auroc", 0.773),
    )
    for environment, model, metric, expected in pooled_claims:
        values = [
            float(row["value"])
            for row in pooled
            if row["environment"] == environment
            and row["model"] == model
            and row["metric"] == metric
        ]
        observed = sum(values) / len(values) if values else float("nan")
        if len(values) != 3 or not math.isfinite(observed) or abs(observed - expected) > tolerance:
            problems.append(f"manuscript pooled-control claim mismatch: {environment}/{model}/{metric}")
        else:
            claim_matches += 1

    return {
        "name": "figure_sources",
        "status": "PASS" if not problems else "FAIL",
        "metric_tables_checked": 2,
        "shuffle_cells": len(shuffle),
        "pooled_cells": len(pooled),
        "manuscript_claims_matched": claim_matches,
        "problems": problems[:20],
    }


def check_metric_level(*, tolerance: float = 5e-4) -> dict[str, Any]:
    """Recompute the metric-level claims and compare them with the manifest."""

    from ppi_shift_release import tables

    result = tables.verify(ROOT, tolerance=tolerance)
    problems = [
        f"{check['claim']}: expected {check['expected']}, observed {check['observed']}"
        for check in result["detail"]
        if check["status"] != "MATCH"
    ]
    if result["checks"] != 56 or result["matched"] != 56:
        problems.append(
            f"expected 56 metric-level checks, found {result['checks']} "
            f"with {result['matched']} matches"
        )
    return {
        "name": "metric_level",
        "status": "PASS" if result["status"] == "OK" else "FAIL",
        "claims_checked": result["checks"],
        "claims_matched": result["matched"],
        "tolerance": tolerance,
        "problems": problems[:20],
    }


def check_release_audit_tables() -> dict[str, Any]:
    """Validate the released aggregate collision, homology, and control tables."""

    base = ROOT / "results/metric_level"
    expected_rows = {
        "known_positive_collision_audit.csv": 96,
        "huri_collision_audit.csv": 24,
        "huri_mapping_coverage.csv": 9,
        "huri_collisions_and_controls.csv": 24,
        "homology_sample_alignment_summary.csv": 4,
        "homology_sample_scores.csv": 24,
        "homology_full_panel_scores.csv": 24,
        "endpoint_pair_concentration.csv": 12,
        "endpoint_frequency_audit.csv": 8,
        "negative_distribution_contrast.csv": 4,
        "additive_baselines.csv": 10,
    }
    problems: list[str] = []
    tables: dict[str, list[dict[str, str]]] = {}
    for name, expected in expected_rows.items():
        path = base / name
        if not path.is_file():
            problems.append(f"missing audit table: {name}")
            continue
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error, UnicodeDecodeError) as error:
            problems.append(f"cannot read audit table {name}: {type(error).__name__}")
            continue
        tables[name] = rows
        if len(rows) != expected:
            problems.append(f"{name}: expected {expected} rows, found {len(rows)}")

    def number(row: dict[str, str], field: str, table: str) -> float | None:
        value = row.get(field, "")
        if value in (None, "", "WITHHELD_PRIVATE_PROVENANCE"):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            problems.append(f"{table}: invalid numeric field {field}")
            return None
        if not math.isfinite(parsed):
            problems.append(f"{table}: non-finite numeric field {field}")
            return None
        return parsed

    collisions = tables.get("known_positive_collision_audit.csv", [])
    source_counts: dict[str, int] = {}
    for row in collisions:
        source = row.get("source", "")
        source_counts[source] = source_counts.get(source, 0) + 1
        if source == "huri":
            if row.get("status") != "SOURCE_UNAVAILABLE":
                problems.append("known_positive_collision_audit.csv: HURI availability changed")
            if row.get("colliding_negative_rows", "") or row.get("collision_fraction", ""):
                problems.append(
                    "known_positive_collision_audit.csv: unavailable HURI shown as measured"
                )
        elif source in {"biogrid", "string", "intact"}:
            if row.get("status") != "MEASURED_POSTHOC":
                problems.append(f"known_positive_collision_audit.csv: invalid status for {source}")
            count = number(row, "colliding_negative_rows", "known_positive_collision_audit.csv")
            fraction = number(row, "collision_fraction", "known_positive_collision_audit.csv")
            denominator = number(row, "negative_rows", "known_positive_collision_audit.csv")
            if count is None or fraction is None or denominator in (None, 0):
                problems.append(
                    f"known_positive_collision_audit.csv: incomplete measured row for {source}"
                )
            elif abs(count / denominator - fraction) > 1e-12:
                problems.append(
                    f"known_positive_collision_audit.csv: fraction mismatch for {source}"
                )
        else:
            problems.append(f"known_positive_collision_audit.csv: unexpected source {source!r}")
    if source_counts != {"biogrid": 24, "string": 24, "intact": 24, "huri": 24}:
        problems.append("known_positive_collision_audit.csv: source row coverage mismatch")

    huri = tables.get("huri_collision_audit.csv", [])
    huri_totals = {
        field: sum(number(row, field, "huri_collision_audit.csv") or 0.0 for row in huri)
        for field in (
            "negative_pair_count",
            "comparable_pair_count",
            "ambiguous_endpoint_pair_count",
            "out_of_universe_pair_count",
            "confirmed_positive_collisions",
        )
    }
    expected_huri = {
        "negative_pair_count": 560382,
        "comparable_pair_count": 185737,
        "ambiguous_endpoint_pair_count": 304,
        "out_of_universe_pair_count": 374341,
        "confirmed_positive_collisions": 0,
    }
    if huri_totals != expected_huri:
        problems.append(f"huri_collision_audit.csv: totals mismatch ({huri_totals})")
    for row in huri:
        negative = number(row, "negative_pair_count", "huri_collision_audit.csv")
        partition = sum(
            number(row, field, "huri_collision_audit.csv") or 0.0
            for field in (
                "comparable_pair_count",
                "ambiguous_endpoint_pair_count",
                "out_of_universe_pair_count",
            )
        )
        if negative != partition or row.get("partition_complete", "").lower() != "true":
            problems.append("huri_collision_audit.csv: incomplete pair partition")

    mapping_rows = tables.get("huri_mapping_coverage.csv", [])
    mapping = {(row.get("level"), row.get("status")): number(
        row, "count", "huri_mapping_coverage.csv"
    ) for row in mapping_rows}
    expected_mapping = {
        ("HuRI genes", "resolved"): 8080,
        ("HuRI genes", "ambiguous"): 7,
        ("HuRI genes", "unmapped"): 158,
        ("HuRI pairs", "resolved"): 50489,
        ("HuRI pairs", "ambiguous"): 34,
        ("HuRI pairs", "unmapped"): 1545,
        ("Benchmark negatives", "comparable"): 185737,
        ("Benchmark negatives", "ambiguous endpoint"): 304,
        ("Benchmark negatives", "out of universe"): 374341,
    }
    if mapping != expected_mapping:
        problems.append("huri_mapping_coverage.csv: mapping and denominator counts mismatch")

    huri_controls = tables.get("huri_collisions_and_controls.csv", [])
    for row in huri_controls:
        partition = sum(
            number(row, field, "huri_collisions_and_controls.csv") or 0.0
            for field in (
                "comparable_pair_count",
                "ambiguous_endpoint_pair_count",
                "out_of_universe_pair_count",
            )
        )
        if number(row, "negative_pair_count", "huri_collisions_and_controls.csv") != partition:
            problems.append("huri_collisions_and_controls.csv: incomplete pair partition")

    alignment = tables.get("homology_sample_alignment_summary.csv", [])
    expected_alignment = {
        ("E003", "esm_partner_matched"): (1988, 166),
        ("E005", "esm_partner_matched"): (1986, 164),
    }
    if len(alignment) == 4:
        for cohort, environment in expected_alignment:
            rows = [
                row
                for row in alignment
                if row.get("cohort") == cohort and row.get("environment") == environment
            ]
            observed = tuple(
                int(number(rows[0], field, "homology_sample_alignment_summary.csv") or -1)
                for field in ("aligned_rows", "identity30_coverage80_rows")
            ) if len(rows) == 1 else None
            if observed != expected_alignment[(cohort, environment)]:
                problems.append("homology_sample_alignment_summary.csv: frozen ESM sample mismatch")
        if any(
            number(row, "sample_rows", "homology_sample_alignment_summary.csv") != 2000
            for row in alignment
        ):
            problems.append("homology_sample_alignment_summary.csv: expected 2,000 sampled rows")

    sample_scores = tables.get("homology_sample_scores.csv", [])
    if len(sample_scores) == 24:
        for row in sample_scores:
            cohort = row.get("cohort")
            expected_measured = 1988 if cohort == "E003" else 1986 if cohort == "E005" else None
            expected_removed = 166 if cohort == "E003" else 164 if cohort == "E005" else None
            if (
                number(row, "filtered_pairs", "homology_sample_scores.csv") != 1822
                or number(row, "measured_pairs", "homology_sample_scores.csv") != expected_measured
                or number(row, "removed_close_pairs", "homology_sample_scores.csv")
                != expected_removed
                or number(row, "filtered_prevalence", "homology_sample_scores.csv") != 0.5
            ):
                problems.append("homology_sample_scores.csv: filtered sample denominator mismatch")
            for observed, reported, field in (
                ("full_auroc", "reported_full_auroc", "AUROC"),
                ("full_auprc", "reported_full_auprc", "AUPRC"),
            ):
                left = number(row, observed, "homology_sample_scores.csv")
                right = number(row, reported, "homology_sample_scores.csv")
                if left is None or right is None or abs(left - right) > 1e-12:
                    problems.append(f"homology_sample_scores.csv: full-panel {field} mismatch")

    full_homology = tables.get("homology_full_panel_scores.csv", [])
    expected_full = {
        "E003": (10051, 9204, 60, 847),
        "E005": (20663, 18741, 155, 1922),
    }
    if len(full_homology) == 24:
        for row in full_homology:
            cohort = row.get("cohort", "")
            observed = tuple(
                int(number(row, field, "homology_full_panel_scores.csv") or -1)
                for field in (
                    "measured_pairs",
                    "filtered_pairs",
                    "unavailable_pairs",
                    "removed_close_pairs",
                )
            )
            if row.get("scope") != "full_panel" or observed != expected_full.get(cohort):
                problems.append("homology_full_panel_scores.csv: frozen full-panel counts mismatch")
            if number(row, "bootstrap_replicates", "homology_full_panel_scores.csv") != 500:
                problems.append(
                    "homology_full_panel_scores.csv: expected 500 anchor-cluster resamples"
                )
            metric_pairs = (
                ("full_auroc", "reported_full_auroc"),
                ("full_auprc", "reported_full_auprc"),
            )
            for observed_field, reported_field in metric_pairs:
                left = number(row, observed_field, "homology_full_panel_scores.csv")
                right = number(row, reported_field, "homology_full_panel_scores.csv")
                if left is None or right is None or abs(left - right) > 1e-12:
                    problems.append(
                        "homology_full_panel_scores.csv: reported full-panel metric mismatch"
                    )

    concentration = tables.get("endpoint_pair_concentration.csv", [])
    expected_concentration = {
        "esm_partner_matched": (996, 172),
        "uniform_partner": (129, 22),
    }
    for environment, expected in expected_concentration.items():
        rows = [
            row
            for row in concentration
            if row.get("cohort") == "E005" and row.get("environment") == environment
        ]
        observed = tuple(
            int(number(rows[0], field, "endpoint_pair_concentration.csv") or -1)
            for field in ("duplicate_negative_pair_rows", "max_candidate_endpoint_reuse")
        ) if len(rows) == 1 else None
        if observed != expected:
            problems.append(f"endpoint_pair_concentration.csv: E005 {environment} counts mismatch")

    endpoint_frequency = tables.get("endpoint_frequency_audit.csv", [])
    for environment, expected in (("uniform_partner", 401), ("fully_random_pair", 51)):
        rows = [
            row for row in endpoint_frequency
            if row.get("stage") == "train"
            and row.get("label") == "0"
            and row.get("environment") == environment
        ]
        observed = (
            number(rows[0], "degree_max", "endpoint_frequency_audit.csv")
            if len(rows) == 1
            else None
        )
        if observed != expected:
            problems.append(
                "endpoint_frequency_audit.csv: training negative degree maximum mismatch "
                f"for {environment}"
            )

    contrasts = tables.get("negative_distribution_contrast.csv", [])
    expected_deltas = {("train", "0"): -350.0, ("validation", "0"): -215.0}
    for key, expected in expected_deltas.items():
        rows = [row for row in contrasts if (row.get("stage"), row.get("label")) == key]
        observed = (
            number(rows[0], "delta_degree_max", "negative_distribution_contrast.csv")
            if len(rows) == 1
            else None
        )
        if observed != expected:
            problems.append(
                f"negative_distribution_contrast.csv: degree maximum delta mismatch for {key}"
            )

    additive = tables.get("additive_baselines.csv", [])
    additive_keys = {(row.get("model"), row.get("metric")) for row in additive}
    if len(additive_keys) != 10 or len(additive) != 10:
        problems.append("additive_baselines.csv: expected unique five-model by two-metric rows")
    for row in additive:
        mean = number(row, "mean", "additive_baselines.csv")
        count = number(row, "n", "additive_baselines.csv")
        if mean is None or not 0.0 <= mean <= 1.0 or count is None or count <= 0:
            problems.append("additive_baselines.csv: invalid metric summary")

    return {
        "name": "release_audit_tables",
        "status": "PASS" if not problems else "FAIL",
        "tables_checked": len(tables),
        "rows_checked": sum(len(rows) for rows in tables.values()),
        "huri_negative_pairs": int(huri_totals["negative_pair_count"]),
        "huri_comparable_pairs": int(huri_totals["comparable_pair_count"]),
        "huri_confirmed_positive_collisions": int(huri_totals["confirmed_positive_collisions"]),
        "problems": problems[:30],
    }


def check_anonymity() -> dict[str, Any]:
    """Scan the text and metadata surfaces for identifying strings.

    ``scripts/validate_release.py`` knows about Git commit identifiers, which is
    exactly the class of leak the first release missed.  It is invoked here so
    that a reviewer running the documented commands also runs the audit. The
    full compressed-payload scan is part of release construction and clean-room
    verification; these modes hash every payload through the integrity manifest.
    """

    report = ROOT / "results/tables/validation_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_release.py"),
            "--root",
            str(ROOT),
            "--skip-integrity",
            "--skip-payloads",
            "--strict-artifacts",
            "--output",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 and not report.exists():
        return {
            "name": "anonymity",
            "status": "FAIL",
            "problems": [completed.stderr.strip() or "scanner did not run"],
        }
    payload = json.loads(report.read_text(encoding="utf-8"))
    findings = payload.get("anonymity_findings", [])
    problems = [
        f"{hit['file']}:{hit['line']} [{hit['kind']}] {hit['text']}"
        for hit in findings[:10]
    ]
    problems.extend(payload.get("problems", [])[:10])
    return {
        "name": "anonymity",
        "status": "PASS" if not problems else "FAIL",
        "findings": len(findings),
        "problems": problems,
    }


def run_figures() -> dict[str, Any]:
    """Regenerate figures in a temporary directory without touching release bytes."""

    import subprocess
    import tempfile

    script = ROOT / "scripts/reproduce_figures.py"
    with tempfile.TemporaryDirectory(prefix=".figure-check-", dir=ROOT) as temporary:
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--root",
                str(ROOT),
                "--output",
                temporary,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "name": "figures",
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "stdout": completed.stdout.strip().splitlines()[-5:],
            "stderr": completed.stderr.strip().splitlines()[-5:],
            "problems": [] if completed.returncode == 0 else ["figure regeneration failed"],
        }


def _report(results: list[dict[str, Any]], *, mode: str, elapsed: float) -> int:
    payload = {
        "mode": mode,
        "elapsed_seconds": round(elapsed, 2),
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL",
        "checks": results,
    }
    output = ROOT / "results/tables/reproduction_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nsequence-benchmark reproduction - mode {mode} - {payload['elapsed_seconds']}s")
    print("-" * 68)
    counts = _coverage_counts()
    print(
        "Coverage inventory: "
        + " ".join(f"{name}={count}" for name, count in counts.items())
    )
    for item in results:
        detail = ""
        for key in (
            "files_checked",
            "panels_verified",
            "pairs_distributed",
            "hash_only",
            "blocked_rights",
            "missing",
            "conditional_rights",
            "source_recoverable_but_not_cleared",
            "cells_matched",
            "mcc_cells_matched",
            "claims_matched",
        ):
            if key in item:
                detail += f" {key}={item[key]}"
        print(f"  {item['status']:4s}  {item['name']}{detail}")
        for problem in item.get("problems", [])[:5]:
            print(f"        - {problem}")
    print("-" * 68)
    print(f"RESULT: {payload['status']}")
    print(f"report: {output.relative_to(ROOT)}")
    return 0 if payload["status"] == "PASS" else 1


def _coverage_counts() -> dict[str, int]:
    """Count each evidence level from the release manifests and shipped files."""

    pointwise = len(list((ROOT / "results/pointwise").rglob("*.csv.gz")))
    metric_level = 0
    hash_only = 0
    source_recoverable = 0
    conditional_rights = 0
    missing = 0
    blocked_rights = 0
    try:
        from ppi_shift_release import tables

        metric_level = int(tables.verify(ROOT)["checks"])
    except (OSError, json.JSONDecodeError):
        missing += 1
    try:
        panel_path = ROOT / "data/manifests/panel_manifest_all44.json"
        panels = json.loads(panel_path.read_text(encoding="utf-8"))["panels"]
        hash_only = sum(panel.get("availability") == "HASH_ONLY" for panel in panels)
        source_recoverable = sum(
            panel.get("availability") == "SOURCE_RECOVERABLE_BUT_NOT_CLEARED"
            for panel in panels
        )
        conditional_rights = sum(
            panel.get("rights_status") == "CONDITIONAL" for panel in panels
        )
        blocked_rights += sum(
            panel.get("availability") == "BLOCKED_RIGHTS"
            or panel.get("rights_status") in {"BLOCKED", "UNKNOWN"}
            for panel in panels
        )
        missing += sum(panel.get("availability") == "MISSING" for panel in panels)
    except (OSError, json.JSONDecodeError, KeyError):
        missing += 1
    try:
        inventory_path = ROOT / "manifests/CHECKPOINT_INVENTORY.csv"
        with inventory_path.open(newline="", encoding="utf-8-sig") as handle:
            blocked_rights += sum(
                row.get("acquisition_status") == "BLOCKED_REDISTRIBUTION_RIGHTS"
                for row in csv.DictReader(handle)
            )
    except OSError:
        missing += 1
    try:
        with (ROOT / "results/tables/reported_metrics.csv").open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            reported_cells = sum(1 for _ in csv.DictReader(handle))
        missing += max(0, reported_cells - pointwise)
    except OSError:
        missing += 1
    return {
        "PREDICTION_LEVEL": pointwise,
        "METRIC_LEVEL": metric_level,
        "HASH_ONLY": hash_only,
        "BLOCKED_RIGHTS": blocked_rights,
        "MISSING": missing,
        "SOURCE_RECOVERABLE_BUT_NOT_CLEARED": source_recoverable,
        "CONDITIONAL_RIGHTS": conditional_rights,
    }


def check_final_assets() -> dict[str, Any]:
    """Check exact current artwork and final inventory requirements."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from verify_final import verify
    try:
        result = verify()
    except (AssertionError, OSError, KeyError, ValueError) as error:
        return {"name": "final_paper_assets", "status": "FAIL", "problems": [repr(error)]}
    return {"name": "final_paper_assets", **result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="fast CPU sanity check")
    mode.add_argument("--all", action="store_true", help="full results reproduction")
    mode.add_argument("--figures", action="store_true", help="regenerate figures only")
    mode.add_argument("--verify", action="store_true", help="integrity check only")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args(argv)

    started = time.time()
    results: list[dict[str, Any]] = []

    if args.smoke:
        results.append(check_environment())
        integrity = check_integrity(limit=40)
        results.append(integrity)
        if integrity["status"] != "PASS":
            return _report(results, mode="smoke", elapsed=time.time() - started)
        results.append(check_panels(limit=2))
        results.append(check_full_panel_manifest())
        results.append(check_pointwise(limit=4, tolerance=args.tolerance))
        results.append(check_mcc(limit=2, tolerance=args.tolerance))
        results.append(check_metric_level())
        results.append(check_release_audit_tables())
        results.append(check_anonymity())
        return _report(results, mode="smoke", elapsed=time.time() - started)

    if args.verify:
        results.append(check_environment())
        integrity = check_integrity()
        results.append(integrity)
        if integrity["status"] != "PASS":
            return _report(results, mode="verify", elapsed=time.time() - started)
        results.append(check_panels())
        results.append(check_full_panel_manifest())
        results.append(check_figure_sources())
        results.append(check_final_assets())
        results.append(check_release_audit_tables())
        results.append(check_anonymity())
        return _report(results, mode="verify", elapsed=time.time() - started)

    if args.figures:
        results.append(run_figures())
        return _report(results, mode="figures", elapsed=time.time() - started)

    results.append(check_environment())
    integrity = check_integrity()
    results.append(integrity)
    if integrity["status"] != "PASS":
        return _report(results, mode="all", elapsed=time.time() - started)
    results.append(check_panels())
    results.append(check_full_panel_manifest())
    results.append(check_pointwise(tolerance=args.tolerance))
    results.append(check_mcc(tolerance=args.tolerance))
    results.append(check_metric_level())
    results.append(check_release_audit_tables())
    results.append(check_figure_sources())
    results.append(check_final_assets())
    results.append(run_figures())
    results.append(check_anonymity())
    return _report(results, mode="all", elapsed=time.time() - started)


if __name__ == "__main__":
    raise SystemExit(main())
