"""Frozen-ESM2 pair MLP: the executed architecture, as a standalone definition.

This is the model that produced the ``symmetric_frozen_esm2_pair_mlp`` rows in
the released tables.  It is an in-house symmetric baseline, not a reproduction
of any published architecture, and the release does not claim parity with one.

The definition is deliberately split from the training loop.  What is shipped
here is the part that determines the function class: the pair-feature
construction and the head.  The training loop, the checkpoint files and the
frozen ESM2 embeddings are not redistributed; see ``models/README.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_ID = "symmetric_frozen_esm2_pair_mlp"
MODEL_VERSION = "ppi_shift.frozen_esm2_pair_mlp.v1"

EMBEDDING_DIM = 640
HIDDEN_LAYERS = (64, 32)
DEFAULT_SEED = 1729


def partner_hash(row: Mapping[str, Any]) -> str:
    """Return the sequence hash of the partner the row actually scores against.

    A positive row scores the curated partner; a negative row scores the
    constructed candidate.  Reading the wrong column is the endpoint-consumption
    error that invalidated the submitted partner-shuffling analysis, so the
    distinction is enforced here rather than left to the caller.
    """

    label = str(row.get("interaction_label", ""))
    if label == "1":
        value = row.get("true_partner_sequence_sha256", "")
    else:
        value = row.get("candidate_sequence_sha256", "")
    if not str(value):
        raise ValueError(f"panel row lacks the selected partner hash: {row.get('row_id')}")
    return str(value)


def symmetric_pair_features(
    rows: Sequence[Mapping[str, Any]], vectors: Mapping[str, Sequence[float]]
) -> np.ndarray:
    """Encode each pair as ``[a+b, |a-b|, a*b]`` over frozen 640-d vectors.

    Every block is invariant under swapping the two proteins, so the model
    cannot use the storage order of a pair.  That is what makes the Fully Random
    arm's lexicographic negative ordering uninformative to this architecture.
    """

    features: list[np.ndarray] = []
    for row in rows:
        anchor = str(row.get("anchor_sequence_sha256", ""))
        partner = partner_hash(row)
        if anchor not in vectors or partner not in vectors:
            raise ValueError(f"embedding coverage missing for row {row.get('row_id')}")
        left = np.asarray(vectors[anchor], dtype=np.float64)
        right = np.asarray(vectors[partner], dtype=np.float64)
        if left.shape != (EMBEDDING_DIM,) or right.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"expected {EMBEDDING_DIM}-dimensional frozen ESM2 means, "
                f"got {left.shape} and {right.shape}"
            )
        features.append(np.concatenate((left + right, np.abs(left - right), left * right)))
    if not features:
        raise ValueError("pair feature extraction requires at least one row")
    return np.asarray(features, dtype=np.float64)


def build_head(*, seed: int = DEFAULT_SEED, max_iter: int = 20) -> Any:
    """Return the executed head: standardise, then a 64/32 ReLU MLP."""

    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYERS,
            activation="relu",
            solver="adam",
            batch_size="auto",
            max_iter=max_iter,
            random_state=seed,
            early_stopping=False,
            tol=0.0,
        ),
    )


def swap_invariance_error(
    rows: Sequence[Mapping[str, Any]], vectors: Mapping[str, Sequence[float]]
) -> float:
    """Maximum absolute feature change under swapping the two endpoints."""

    direct = symmetric_pair_features(rows, vectors)
    swapped_rows = []
    for row in rows:
        flipped = dict(row)
        flipped["anchor_sequence_sha256"] = row.get("true_partner_sequence_sha256", "")
        flipped["true_partner_sequence_sha256"] = row.get("anchor_sequence_sha256", "")
        flipped["candidate_sequence_sha256"] = row.get("anchor_sequence_sha256", "")
        swapped_rows.append(flipped)
    swapped = symmetric_pair_features(swapped_rows, vectors)
    return float(np.max(np.abs(direct - swapped)))
