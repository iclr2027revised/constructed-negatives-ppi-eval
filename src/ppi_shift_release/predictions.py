"""Validate an external prediction file against a frozen panel and score it.

The contract is strict on purpose.  A prediction file that is missing rows,
carries duplicates, references an unknown row, or contains a score outside
``[0, 1]`` is rejected rather than silently repaired, because a silently repaired
file produces a number that cannot be compared with the reported tables.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import metrics
from .panels import PanelEntry, open_text, panel_positions

REQUIRED_COLUMNS = ("row_id", "score")


@dataclass(frozen=True)
class PredictionTable:
    row_ids: tuple[str, ...]
    scores: np.ndarray
    labels: np.ndarray


def load_predictions(path: Path, *, score_column: str = "score") -> PredictionTable:
    row_ids: list[str] = []
    scores: list[float] = []
    with open_text(path) as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Prediction file is empty: {path}")
        for column in ("row_id", score_column):
            if column not in header:
                raise ValueError(f"Prediction file lacks column {column!r}: {path}")
        row_id_idx = header.index("row_id")
        score_idx = header.index(score_column)
        for index, row in enumerate(reader, start=2):
            row_id = row[row_id_idx].strip()
            if not row_id:
                raise ValueError(f"{path}:{index}: empty row_id")
            try:
                score = float(row[score_idx])
            except (TypeError, ValueError, IndexError) as error:
                raise ValueError(f"{path}:{index}: score is not a number") from error
            if not np.isfinite(score):
                raise ValueError(f"{path}:{index}: score is not finite")
            if score < 0.0 or score > 1.0:
                raise ValueError(
                    f"{path}:{index}: score {score} is outside [0, 1]. The frozen convention "
                    "expects a probability; apply your own calibration before submitting."
                )
            row_ids.append(row_id)
            scores.append(score)
    if not row_ids:
        raise ValueError(f"Prediction file is empty: {path}")
    duplicates = len(row_ids) - len(set(row_ids))
    if duplicates:
        raise ValueError(f"{path}: {duplicates} duplicate row_id values")
    return PredictionTable(
        row_ids=tuple(row_ids),
        scores=np.asarray(scores, dtype=np.float64),
        labels=np.zeros(len(row_ids), dtype=np.int64),
    )


def join_to_panel(
    root: Path, entry: PanelEntry, table: PredictionTable
) -> PredictionTable:
    """Align predictions to the frozen panel order and verify coverage."""
    panel_row_ids, position, panel_labels_array = panel_positions(root, entry)
    if table.row_ids == panel_row_ids:
        return PredictionTable(
            row_ids=panel_row_ids,
            scores=table.scores,
            labels=panel_labels_array,
        )
    unknown = [row_id for row_id in table.row_ids if row_id not in position]
    if unknown:
        raise ValueError(
            f"{len(unknown)} predicted row_id values are absent from {entry.path}; "
            f"first: {unknown[0]}"
        )
    scores = np.full(len(panel_row_ids), np.nan, dtype=np.float64)
    indices = np.fromiter(
        (position[row_id] for row_id in table.row_ids),
        dtype=np.int64,
        count=len(table.row_ids),
    )
    scores[indices] = table.scores
    missing = len(panel_row_ids) - len(table.row_ids)
    if missing:
        raise ValueError(
            f"{missing} of {len(panel_row_ids)} frozen panel rows have no prediction. "
            "Partial coverage is rejected; score the whole panel."
        )
    return PredictionTable(
        row_ids=panel_row_ids,
        scores=scores,
        labels=panel_labels_array,
    )


def evaluate(
    root: Path,
    entry: PanelEntry,
    table: PredictionTable,
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 1729,
) -> dict[str, Any]:
    joined = join_to_panel(root, entry, table)
    summary = metrics.summarise(
        joined.labels,
        joined.scores,
        score_convention=entry.score_convention,
    )
    summary.update(
        {
            "cohort": entry.cohort,
            "split": entry.split,
            "environment": entry.environment,
            "panel_path": entry.path,
            "panel_sha256": entry.sha256,
            "evidence_semantics": entry.evidence_semantics,
            "prevalence_baseline_auprc": metrics.prevalence_baseline(joined.labels),
            "auprc_minus_prevalence": summary["auprc"]
            - metrics.prevalence_baseline(joined.labels),
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(bootstrap_seed),
        }
    )
    return summary


def evaluate_against_reference(
    root: Path,
    entry: PanelEntry,
    table: PredictionTable,
    reference_scores: Sequence[float],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 1729,
) -> dict[str, Any]:
    """Paired comparison of a new model against a released reference model."""
    joined = join_to_panel(root, entry, table)
    reference = np.asarray(reference_scores, dtype=np.float64)
    if reference.size != joined.scores.size:
        raise ValueError("Reference scores do not cover the panel")
    result = evaluate(
        root,
        entry,
        table,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    for metric in ("auroc", "auprc"):
        result[f"paired_vs_reference_{metric}"] = metrics.paired_row_bootstrap(
            joined.labels,
            reference,
            joined.scores,
            metric=metric,
            score_convention=entry.score_convention,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
    return result


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("No metric rows to write")
    fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
