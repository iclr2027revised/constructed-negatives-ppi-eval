"""Metrics with the frozen cohort-specific score and tie conventions.

E003 corrected-replay AUROC/AUPRC clip probabilities to ``[1e-7, 1-1e-7]``
before ranking. E005 primary AUROC/AUPRC use the saved raw probabilities.
Within either convention, sklearn resolves ties; this module adds no jitter or
pre-sort. Calibration metrics keep their documented clipping rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

CALIBRATION_EPSILON = 1e-7
RAW_SCORE_CONVENTION = "raw_probability"
CLIPPED_SCORE_CONVENTION = "probability_clip_1e-7"


def score_convention_for_cohort(cohort: str) -> str:
    """Return the verified historical ranking convention for a cohort."""

    if cohort == "E003_corrected_replay":
        return CLIPPED_SCORE_CONVENTION
    return RAW_SCORE_CONVENTION


def _ranking_scores(scores: np.ndarray, score_convention: str) -> np.ndarray:
    if score_convention == RAW_SCORE_CONVENTION:
        return scores
    if score_convention == CLIPPED_SCORE_CONVENTION:
        return np.clip(scores, CALIBRATION_EPSILON, 1.0 - CALIBRATION_EPSILON)
    raise ValueError(f"Unknown score convention: {score_convention!r}")


def _aligned(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.size != scores_array.size:
        raise ValueError("Labels and scores must have the same length")
    if labels_array.size == 0:
        raise ValueError("Labels and scores must be non-empty")
    if not np.isfinite(scores_array).all():
        raise ValueError("Scores contain non-finite values")
    if not set(np.unique(labels_array).tolist()) <= {0, 1}:
        raise ValueError("Labels must be binary 0/1")
    return labels_array, scores_array


def auroc(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    score_convention: str = RAW_SCORE_CONVENTION,
) -> float:
    labels_array, scores_array = _aligned(labels, scores)
    if labels_array.min() == labels_array.max():
        raise ValueError("AUROC is undefined when only one class is present")
    return float(roc_auc_score(labels_array, _ranking_scores(scores_array, score_convention)))


def auprc(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    score_convention: str = RAW_SCORE_CONVENTION,
) -> float:
    labels_array, scores_array = _aligned(labels, scores)
    if labels_array.min() == labels_array.max():
        raise ValueError("AUPRC is undefined when only one class is present")
    return float(
        average_precision_score(labels_array, _ranking_scores(scores_array, score_convention))
    )


def mcc_at_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float,
    score_convention: str = RAW_SCORE_CONVENTION,
) -> float:
    labels_array, scores_array = _aligned(labels, scores)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    ranking_scores = _ranking_scores(scores_array, score_convention)
    predictions = (ranking_scores >= threshold).astype(np.int64)
    return float(matthews_corrcoef(labels_array, predictions))


def prevalence_baseline(labels: Sequence[int]) -> float:
    """AUPRC of a constant predictor equals the positive prevalence."""
    labels_array = np.asarray(labels, dtype=np.int64)
    if labels_array.size == 0:
        raise ValueError("Labels must be non-empty")
    return float(labels_array.mean())


def brier(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_array, scores_array = _aligned(labels, scores)
    clipped = np.clip(scores_array, 0.0, 1.0)
    return float(brier_score_loss(labels_array, clipped))


def nll(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_array, scores_array = _aligned(labels, scores)
    clipped = np.clip(scores_array, CALIBRATION_EPSILON, 1.0 - CALIBRATION_EPSILON)
    return float(log_loss(labels_array, clipped, labels=[0, 1]))


def expected_calibration_error(
    labels: Sequence[int], scores: Sequence[float], *, bins: int = 10
) -> float:
    labels_array, scores_array = _aligned(labels, scores)
    clipped = np.clip(scores_array, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (clipped >= lower) & (
            (clipped < upper) if upper < 1.0 else (clipped <= upper)
        )
        if selected.any():
            error += float(selected.mean()) * abs(
                float(labels_array[selected].mean()) - float(clipped[selected].mean())
            )
    return float(error)


def summarise(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    score_convention: str = RAW_SCORE_CONVENTION,
) -> dict[str, Any]:
    """Every metric the release reports for one evaluation surface."""
    return {
        "n_rows": int(len(labels)),
        "n_positive": int(np.sum(np.asarray(labels, dtype=np.int64) == 1)),
        "n_negative": int(np.sum(np.asarray(labels, dtype=np.int64) == 0)),
        "prevalence": prevalence_baseline(labels),
        "auroc": auroc(labels, scores, score_convention=score_convention),
        "auprc": auprc(labels, scores, score_convention=score_convention),
        "score_convention": score_convention,
        "brier": brier(labels, scores),
        "nll": nll(labels, scores),
        "ece": expected_calibration_error(labels, scores),
    }


def paired_row_bootstrap(
    labels: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    metric: str = "auroc",
    score_convention: str = RAW_SCORE_CONVENTION,
    replicates: int = 2000,
    seed: int = 1729,
) -> dict[str, float]:
    """Percentile interval for ``metric(b) - metric(a)`` resampling rows jointly.

    The two score vectors are resampled with the same draw, so the interval
    describes the paired difference rather than two independent metrics.
    """
    labels_array, scores_a_array = _aligned(labels, scores_a)
    _, scores_b_array = _aligned(labels, scores_b)
    function = {"auroc": auroc, "auprc": auprc}[metric]
    rng = np.random.default_rng(seed)
    n = labels_array.size
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        index = rng.integers(0, n, n)
        sampled_labels = labels_array[index]
        if sampled_labels.min() == sampled_labels.max():
            deltas[replicate] = np.nan
            continue
        deltas[replicate] = function(
            sampled_labels, scores_b_array[index], score_convention=score_convention
        ) - function(
            sampled_labels, scores_a_array[index], score_convention=score_convention
        )
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        raise ValueError("Every bootstrap replicate was degenerate")
    lower, upper = np.percentile(finite, (2.5, 97.5))
    return {
        "metric": metric,
        "point": function(
            labels_array, scores_b_array, score_convention=score_convention
        )
        - function(labels_array, scores_a_array, score_convention=score_convention),
        "lower": float(lower),
        "upper": float(upper),
        "replicates": int(replicates),
        "valid_replicates": int(finite.size),
        "seed": int(seed),
    }
