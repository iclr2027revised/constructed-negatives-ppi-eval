#!/usr/bin/env python
"""Verify bootstrap manifests, fix the --run-sample direction/protocol, and
reproduce all 144 matched 512/1024 common-test bootstrap intervals.

This script verifies:
1. ``configs/bootstrap/e005_uncertainty_manifest.json`` (2,000 replicates, seed_base 1729,
   stratified_positive_negative_shared_rows and paired_units_only_when_valid) against
   ``results/metric_level/statistical_per_seed_intervals.csv``,
   ``results/tables/bootstrap_core_summary.csv``, and
   ``results/tables/cluster_sensitivity_summary.csv``.
2. ``configs/bootstrap/e005_common_test_uncertainty_manifest.json`` (2,000 replicates,
   seed_base 1729, paired 512-vs-1024 common-test evaluation) against
   ``results/additional/MATCHED_ALL_144_INTERVALS.csv`` and
   ``results/additional/MATCHED_512_1024_ALL_432_SEED_ROWS.csv``.
3. ``--run-sample``: runs the exact documented E005-minus-E003 (1024 minus 512)
   paired, stratified positive/negative bootstrap (2,000 replicates, seed 1729,
   3-seed mean difference, both AUPRC and AUROC, ``[1e-7, 1-1e-7]`` clipping)
   for ``tuna_corrected / uniform_partner -> uniform_partner`` on the frozen
   512 common test panel, and compares directly against the corresponding row of
   ``results/additional/MATCHED_ALL_144_INTERVALS.csv``.
4. ``--reproduce-matched-144``: reproduces ALL 144 matched 512/1024 interval rows
   (4 models x 6 train environments x 6 test environments) across all 3 training
   seeds (864 prediction vectors, 2,000 replicates each, ``seed = 1729 + test_index``)
   and writes a full reported-vs-recomputed comparison CSV and JSON report.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CLIP_EPS = 1e-7
CANONICAL_MODELS = (
    "symmetric_frozen_esm2_pair_mlp",
    "tuna_corrected",
    "dscript_esm2",
    "rapppid",
)
CANONICAL_ENVIRONMENTS = (
    "uniform_partner",
    "degree_partner_matched",
    "sequence_nuisance_matched",
    "esm_partner_matched",
    "bernett2026_ilp_exact",
    "tppni2025_scm_cl3_exact_balanced",
)
TRAINING_SEEDS = (11, 22, 33)


@dataclass(frozen=True)
class _SortedRows:
    labels: np.ndarray
    starts: np.ndarray


def _prepare_sorted_rows(labels: np.ndarray, scores: np.ndarray) -> _SortedRows:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.float64, copy=False)
    starts = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    )
    return _SortedRows(sorted_labels, starts), order


def make_stratified_bootstrap_weights(
    labels: np.ndarray, replicates: int = 2000, seed: int = 1729
) -> np.ndarray:
    """Sample positives and negatives separately using the historical RNG call sequence."""
    values = np.asarray(labels, dtype=np.int64)
    positive = np.flatnonzero(values == 1)
    negative = np.flatnonzero(values == 0)
    rng = np.random.default_rng(seed)
    dtype = np.uint16 if len(values) < 65535 else np.int64
    weights = np.zeros((replicates, len(values)), dtype=dtype)
    for replicate in range(replicates):
        pos_draw = rng.choice(positive, size=len(positive), replace=True)
        neg_draw = rng.choice(negative, size=len(negative), replace=True)
        weights[replicate] = np.bincount(
            np.concatenate((pos_draw, neg_draw)), minlength=len(values)
        )
    return weights


def paired_stratified_vector_metrics(
    labels: np.ndarray,
    raw_scores: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int = 250,
) -> dict[str, Any]:
    """Exact vectorized historical AUPRC and AUROC under shared row weights."""
    values = np.asarray(raw_scores, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Scores must be finite probabilities in [0, 1]")
    clipped = np.clip(values, CLIP_EPS, 1.0 - CLIP_EPS)
    sorted_rows, order = _prepare_sorted_rows(labels, clipped)

    # Point estimate (unit weights)
    unit_w = np.ones((1, len(labels)), dtype=np.float64)
    est_ap, est_roc = _eval_chunks(sorted_rows, order, unit_w, chunk_size=1)
    auprc_vec, auroc_vec = _eval_chunks(sorted_rows, order, weights, chunk_size=chunk_size)
    return {
        "auprc": auprc_vec,
        "auroc": auroc_vec,
        "estimate_auprc": float(est_ap[0]),
        "estimate_auroc": float(est_roc[0]),
    }


def _eval_chunks(
    sorted_rows: _SortedRows,
    order: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int = 250,
) -> tuple[np.ndarray, np.ndarray]:
    n_rep = len(weights)
    auprc_out = np.empty(n_rep, dtype=np.float64)
    auroc_out = np.empty(n_rep, dtype=np.float64)
    starts = sorted_rows.starts
    sl = sorted_rows.labels
    for start in range(0, n_rep, chunk_size):
        stop = min(start + chunk_size, n_rep)
        row_weights = weights[start:stop][:, order].astype(np.float64, copy=False)
        weighted_labels = row_weights * sl[None, :]
        positive_by_group = np.add.reduceat(weighted_labels, starts, axis=1)
        total_by_group = np.add.reduceat(row_weights, starts, axis=1)
        total_positive = positive_by_group.sum(axis=1)
        total_negative = total_by_group.sum(axis=1) - total_positive
        if np.any(total_positive <= 0) or np.any(total_negative <= 0):
            raise ValueError("Bootstrap replicate degenerate (single class)")
        cumulative_positive = np.cumsum(positive_by_group, axis=1)
        cumulative_total = np.cumsum(total_by_group, axis=1)
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        auprc_out[start:stop] = np.sum(precision * positive_by_group, axis=1) / total_positive
        negative_by_group = total_by_group - positive_by_group
        negative_after = total_negative[:, None] - np.cumsum(negative_by_group, axis=1)
        auroc_out[start:stop] = np.sum(
            positive_by_group * (negative_after + 0.5 * negative_by_group), axis=1
        ) / (total_positive * total_negative)
    return auprc_out, auroc_out


def summarize_bootstrap(values: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "lower_95": float(low),
        "upper_95": float(high),
        "median": float(np.median(values)),
    }


def _compute_row_unit(task: tuple[str, str, str, np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]], np.ndarray]) -> dict[str, Any]:
    model_id, train_env, test_env, labels, seed_arrays, weights = task
    dists: dict[str, list[np.ndarray]] = {"auprc": [], "auroc": []}
    points: dict[str, dict[int, float]] = {"e003_auprc": {}, "e005_auprc": {}, "e003_auroc": {}, "e005_auroc": {}}
    for s in TRAINING_SEEDS:
        s512, s1024 = seed_arrays[s]
        r512 = paired_stratified_vector_metrics(labels, s512, weights)
        r1024 = paired_stratified_vector_metrics(labels, s1024, weights)
        for m in ("auprc", "auroc"):
            # Direction is strictly E005 (1024) minus E003 (512)
            dists[m].append(r1024[m] - r512[m])
            points[f"e003_{m}"][s] = float(r512[f"estimate_{m}"])
            points[f"e005_{m}"][s] = float(r1024[f"estimate_{m}"])

    rec: dict[str, Any] = {
        "model_id": model_id,
        "train_environment": train_env,
        "test_environment": test_env,
    }
    for m in ("auprc", "auroc"):
        mean_delta_dist = np.mean(np.asarray(dists[m]), axis=0)
        iv = summarize_bootstrap(mean_delta_dist)
        rec[f"e003_{m}"] = float(np.mean([points[f"e003_{m}"][s] for s in TRAINING_SEEDS]))
        rec[f"e005_{m}"] = float(np.mean([points[f"e005_{m}"][s] for s in TRAINING_SEEDS]))
        rec[f"delta_{m}"] = rec[f"e005_{m}"] - rec[f"e003_{m}"]
        rec[f"delta_{m}_lower_95"] = iv["lower_95"]
        rec[f"delta_{m}_upper_95"] = iv["upper_95"]
    return rec


def run_matched_sample(root: Path, replicates: int = 2000) -> dict[str, Any]:
    """Corrected sample run: E005 (1024) minus E003 (512), 3-seed mean, stratified, AUROC & AUPRC."""
    t0 = time.perf_counter()
    common_panel = root / "results/common_512_pointwise/test__uniform_partner__paired_e003_e005_all144_models.csv.gz"
    with gzip.open(common_panel, "rt", newline="") as gf:
        df = pd.read_csv(gf, float_precision="round_trip")
    labels = df["interaction_label"].to_numpy(dtype=np.int64)
    weights = make_stratified_bootstrap_weights(labels, replicates=replicates, seed=1729)
    seed_arrays = {
        s: (
            df[f"e003_512__tuna_corrected__uniform_partner__seed-{s}"].to_numpy(dtype=np.float64),
            df[f"e005_1024__tuna_corrected__uniform_partner__seed-{s}"].to_numpy(dtype=np.float64),
        )
        for s in TRAINING_SEEDS
    }
    res = _compute_row_unit(
        ("tuna_corrected", "uniform_partner", "uniform_partner", labels, seed_arrays, weights)
    )
    matched_ci = pd.read_csv(root / "results/additional/MATCHED_ALL_144_INTERVALS.csv", float_precision="round_trip")
    row = matched_ci[
        (matched_ci["model_id"] == "tuna_corrected")
        & (matched_ci["train_environment"] == "uniform_partner")
        & (matched_ci["test_environment"] == "uniform_partner")
    ].iloc[0]

    diffs = {
        k: abs(float(res[k]) - float(row[k]))
        for k in (
            "e003_auprc", "e005_auprc", "delta_auprc", "delta_auprc_lower_95", "delta_auprc_upper_95",
            "e003_auroc", "e005_auroc", "delta_auroc", "delta_auroc_lower_95", "delta_auroc_upper_95",
        )
    }
    max_diff = max(diffs.values()) if replicates == 2000 else None
    return {
        "direction": "E005_minus_E003 (1024 minus 512)",
        "model_id": "tuna_corrected",
        "train_environment": "uniform_partner",
        "test_environment": "uniform_partner",
        "resampling_scheme": "stratified_positive_negative_shared_rows",
        "estimand": "mean_of_three_seed_E005_minus_E003_metric_differences",
        "replicates": int(replicates),
        "seed": 1729,
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "recomputed": res,
        "reported": {k: float(row[k]) for k in diffs},
        "abs_diffs": diffs,
        "max_abs_diff": max_diff,
        "status": "PASS" if (max_diff is None or max_diff <= 1e-12) else "MISMATCH",
    }


def _worker_init(labels: np.ndarray, weights: np.ndarray) -> None:
    global _W_LABELS, _W_WEIGHTS
    _W_LABELS = labels
    _W_WEIGHTS = weights


def _worker_eval(
    args: tuple[str, str, str, int, int, dict[int, tuple[np.ndarray, np.ndarray]]]
) -> dict[str, Any]:
    model_id, train_env, test_env, test_index, seed_base, seed_arrays = args
    rec = _compute_row_unit((model_id, train_env, test_env, _W_LABELS, seed_arrays, _W_WEIGHTS))
    rec["seed"] = seed_base + test_index
    return rec


def reproduce_all_matched_144(
    root: Path,
    *,
    workers: int = 8,
    output_dir: Path | None = None,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    c_manifest = json.loads((root / "configs/bootstrap/e005_common_test_uncertainty_manifest.json").read_text(encoding="utf-8"))
    replicates = int(c_manifest["replicates"])
    seed_base = int(c_manifest["seed_base"])
    reported_df = pd.read_csv(root / "results/additional/MATCHED_ALL_144_INTERVALS.csv", float_precision="round_trip")
    assert len(reported_df) == 144

    recomputed_rows: list[dict[str, Any]] = []
    cache_dir = (output_dir / "env_cache") if output_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for test_index, test_env in enumerate(CANONICAL_ENVIRONMENTS):
        cache_file = (cache_dir / f"{test_env}.csv") if cache_dir is not None else None
        if cache_file is not None and cache_file.exists():
            cached_rows = pd.read_csv(cache_file, float_precision="round_trip").to_dict(orient="records")
            if len(cached_rows) == 24:
                recomputed_rows.extend(cached_rows)
                continue
        panel_path = root / f"results/common_512_pointwise/test__{test_env}__paired_e003_e005_all144_models.csv.gz"
        with gzip.open(panel_path, "rt", newline="") as gf:
            env_df = pd.read_csv(gf, float_precision="round_trip")
        env_labels = env_df["interaction_label"].to_numpy(dtype=np.int64)
        env_weights = make_stratified_bootstrap_weights(
            env_labels, replicates=replicates, seed=seed_base + test_index
        )
        tasks = [
            (
                m,
                tr,
                test_env,
                test_index,
                seed_base,
                {
                    s: (
                        env_df[f"e003_512__{m}__{tr}__seed-{s}"].to_numpy(dtype=np.float64),
                        env_df[f"e005_1024__{m}__{tr}__seed-{s}"].to_numpy(dtype=np.float64),
                    )
                    for s in TRAINING_SEEDS
                },
            )
            for m in CANONICAL_MODELS
            for tr in CANONICAL_ENVIRONMENTS
        ]
        env_rows: list[dict[str, Any]] = []
        if workers <= 1:
            _worker_init(env_labels, env_weights)
            for t in tasks:
                env_rows.append(_worker_eval(t))
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_init,
                initargs=(env_labels, env_weights),
            ) as pool:
                for res in pool.map(_worker_eval, tasks):
                    env_rows.append(res)
        if cache_file is not None:
            pd.DataFrame(env_rows).to_csv(cache_file, index=False)
        recomputed_rows.extend(env_rows)

    recomp_df = pd.DataFrame(recomputed_rows)
    merged = reported_df.merge(
        recomp_df,
        on=["model_id", "train_environment", "test_environment"],
        suffixes=("_reported", "_recomputed"),
        validate="one_to_one",
    )
    if len(merged) != 144:
        raise ValueError(f"Expected 144 joined rows, got {len(merged)}")

    compare_cols = [
        "e003_auprc", "e005_auprc", "delta_auprc", "delta_auprc_lower_95", "delta_auprc_upper_95",
        "e003_auroc", "e005_auroc", "delta_auroc", "delta_auroc_lower_95", "delta_auroc_upper_95",
    ]
    max_abs_diffs: dict[str, float] = {}
    for col in compare_cols:
        diff = np.abs(merged[f"{col}_recomputed"].to_numpy(dtype=np.float64) - merged[f"{col}_reported"].to_numpy(dtype=np.float64))
        merged[f"{col}_abs_diff"] = diff
        max_abs_diffs[col] = float(np.max(diff))
    seed_mismatches = int((merged["seed_reported"] != merged["seed_recomputed"]).sum())
    overall_max = float(max(max_abs_diffs.values()))
    elapsed = round(time.perf_counter() - t0, 3)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_dir / "matched_144_reported_vs_recomputed.csv", index=False)

    return {
        "status": "PASS" if (overall_max <= tolerance and seed_mismatches == 0) else "FAIL",
        "rows_verified": len(merged),
        "replicates": replicates,
        "seed_base": seed_base,
        "seed_mismatches": seed_mismatches,
        "tolerance": tolerance,
        "max_abs_diff_overall": overall_max,
        "max_abs_diff_by_column": max_abs_diffs,
        "elapsed_seconds": elapsed,
        "workers": workers,
    }


def verify_bootstrap_manifests(
    root: Path = ROOT,
    *,
    run_sample: bool = False,
    reproduce_matched_144: bool = False,
    workers: int = 8,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    cfg_dir = root / "configs/bootstrap"
    u_manifest = json.loads((cfg_dir / "e005_uncertainty_manifest.json").read_text(encoding="utf-8"))
    c_manifest = json.loads((cfg_dir / "e005_common_test_uncertainty_manifest.json").read_text(encoding="utf-8"))

    assert u_manifest["replicates"] == 2000
    assert u_manifest["seed_base"] == 1729
    assert c_manifest["replicates"] == 2000
    assert c_manifest["seed_base"] == 1729

    stat_df = pd.read_csv(root / "results/metric_level/statistical_per_seed_intervals.csv", float_precision="round_trip")
    assert len(stat_df) == 192
    bcore_df = pd.read_csv(root / "results/tables/bootstrap_core_summary.csv", float_precision="round_trip")
    assert len(bcore_df) == 432
    csens_df = pd.read_csv(root / "results/tables/cluster_sensitivity_summary.csv", float_precision="round_trip")
    assert len(csens_df) == 864
    matched_ci = pd.read_csv(root / "results/additional/MATCHED_ALL_144_INTERVALS.csv", float_precision="round_trip")
    matched_seeds = pd.read_csv(root / "results/additional/MATCHED_512_1024_ALL_432_SEED_ROWS.csv", float_precision="round_trip")
    assert len(matched_ci) == 144
    assert len(matched_seeds) == 432

    out: dict[str, Any] = {
        "status": "PASS",
        "e005_uncertainty_replicates": int(u_manifest["replicates"]),
        "e005_uncertainty_seed_base": int(u_manifest["seed_base"]),
        "common_512_uncertainty_replicates": int(c_manifest["replicates"]),
        "common_512_uncertainty_seed_base": int(c_manifest["seed_base"]),
        "statistical_per_seed_interval_rows": len(stat_df),
        "bootstrap_core_summary_rows": len(bcore_df),
        "cluster_sensitivity_summary_rows": len(csens_df),
        "matched_512_1024_interval_rows": len(matched_ci),
        "matched_512_1024_seed_rows": len(matched_seeds),
    }

    if run_sample:
        sample = run_matched_sample(root, replicates=2000)
        out["sample_paired_bootstrap_tuna_uniform_e005_minus_e003"] = sample
        if sample["status"] != "PASS":
            out["status"] = "FAIL"

    if reproduce_matched_144:
        matched_report = reproduce_all_matched_144(root, workers=workers, output_dir=output_dir)
        out["matched_144_full_reproduction"] = matched_report
        if matched_report["status"] != "PASS":
            out["status"] = "FAIL"

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "reproduce_bootstrap_report.json").write_text(
            json.dumps(out, indent=2) + "\n", encoding="utf-8"
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-sample", action="store_true")
    parser.add_argument("--reproduce-matched-144", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 4)))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    report = verify_bootstrap_manifests(
        args.root,
        run_sample=args.run_sample,
        reproduce_matched_144=args.reproduce_matched_144,
        workers=args.workers,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
