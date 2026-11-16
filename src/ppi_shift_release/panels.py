"""Load frozen panels and resolve their sequences from the released registry.

The released panels carry sequence SHA256 hashes rather than raw sequences.  The
sequence registry carries the sequences.  Joining the two reconstructs every
panel row, and the join is verified rather than assumed: each sequence is
re-hashed and compared with the hash the panel stores.
"""

from __future__ import annotations

import csv
import functools
import gzip
import hashlib
import json
import numpy as np
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PANEL_MANIFEST = "data/manifests/panel_manifest.json"
SEQUENCE_REGISTRY = "data/sequence_registry/sequences.fasta.gz"

LEFT_HASH = "anchor_sequence_sha256"
RIGHT_HASH = "true_partner_sequence_sha256"
CANDIDATE_HASH = "candidate_sequence_sha256"
BROKEN_CANDIDATE_HASH = "broken_candidate_sequence_sha256"

NEGATIVE_ENVIRONMENTS = (
    "uniform_partner",
    "degree_partner_matched",
    "esm_partner_matched",
    "sequence_nuisance_matched",
    "tppni2025_scm_cl3_exact_balanced",
    "bernett2026_ilp_exact",
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open(newline="")


def read_rows(path: Path) -> list[dict[str, str]]:
    with open_text(path) as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class PanelEntry:
    cohort: str
    split: str
    environment: str
    path: str
    sha256: str
    rows: int
    n_positive: int
    n_negative: int
    evidence_semantics: str
    score_convention: str


def load_manifest(root: Path) -> list[PanelEntry]:
    path = root / PANEL_MANIFEST
    document = json.loads(path.read_text(encoding="utf-8"))
    entries = []
    for record in document["panels"]:
        entries.append(
            PanelEntry(
                cohort=record["cohort"],
                split=record["split"],
                environment=record["environment"],
                path=record["path"],
                sha256=record["sha256"],
                rows=int(record["rows"]),
                n_positive=int(record["n_positive"]),
                n_negative=int(record["n_negative"]),
                evidence_semantics=record["evidence_semantics"],
                score_convention=record.get(
                    "score_convention",
                    "probability_clip_1e-7"
                    if record["cohort"] == "E003_corrected_replay"
                    else "raw_probability",
                ),
            )
        )
    return entries


def load_sequences(root: Path) -> dict[str, str]:
    """Return ``sha256 -> sequence`` for the released registry."""
    path = root / SEQUENCE_REGISTRY
    sequences: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    sequences[name] = "".join(chunks)
                name = line[1:].strip()
                chunks = []
            elif line:
                chunks.append(line)
    if name is not None:
        sequences[name] = "".join(chunks)
    return sequences


def verify_sequence_registry(root: Path) -> dict[str, Any]:
    """Every registry entry must hash to the key it is stored under."""
    sequences = load_sequences(root)
    mismatched = [
        name for name, sequence in sequences.items() if sha256_text(sequence) != name
    ]
    return {
        "sequences": len(sequences),
        "mismatched": len(mismatched),
        "mismatched_examples": mismatched[:5],
        "total_residues": sum(len(sequence) for sequence in sequences.values()),
    }


def iter_panel_rows(root: Path, entry: PanelEntry) -> Iterator[dict[str, str]]:
    yield from read_rows(root / entry.path)


def scored_right_endpoint(row: Mapping[str, str], label: int) -> tuple[str, str]:
    """Return ``(sequence_sha256, role)`` for the right endpoint the model scored.

    Broken-pair panels shift the right endpoint of **every** row, positives
    included, and record the shifted endpoint in ``broken_candidate_*``.  The
    ``true_partner_*`` columns keep the original partner, which is not what was
    scored, so a resolver that trusted them would hand an external user the wrong
    sequence for every E005 positive.

    Non-broken panels score the true partner for a positive and the candidate for
    a negative.  ``true_partner_id`` on a negative names the positive the negative
    was derived from, not the protein that was scored.
    """
    broken = row.get(BROKEN_CANDIDATE_HASH, "")
    if broken:
        return broken, "broken_candidate_partner"
    if label == 1:
        return row[RIGHT_HASH], "true_partner"
    if CANDIDATE_HASH not in row:
        raise KeyError(
            f"negative row {row.get('row_id', '?')} has no {CANDIDATE_HASH}; "
            "the scored right endpoint cannot be recovered"
        )
    return row[CANDIDATE_HASH], "candidate_partner"


def resolve_panel(
    root: Path, entry: PanelEntry, sequences: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Return panel rows with both scored endpoint sequences attached.

    The right endpoint depends on the panel family and the label; see
    :func:`scored_right_endpoint`.  Attaching the wrong one would return a protein
    the model never scored.

    Raises if a row references a hash the registry does not carry, or if the stored
    hash does not match the sequence it resolves to.
    """
    resolved: list[dict[str, Any]] = []
    for row in iter_panel_rows(root, entry):
        label = int(row["interaction_label"])
        left_hash = row[LEFT_HASH]
        right_hash, right_role = scored_right_endpoint(row, label)
        for name, digest in (("left", left_hash), ("right", right_hash)):
            if digest not in sequences:
                raise KeyError(
                    f"{entry.path}: {name} sequence {digest} is absent from the registry. "
                    "The release is incomplete; do not substitute a placeholder."
                )
            if sha256_text(sequences[digest]) != digest:
                raise ValueError(
                    f"{entry.path}: {name} sequence does not hash to {digest}"
                )
        resolved.append(
            {
                "row_id": row["row_id"],
                "label": label,
                "left_sequence": sequences[left_hash],
                "right_sequence": sequences[right_hash],
                "left_sha256": left_hash,
                "right_sha256": right_hash,
                "right_endpoint_role": right_role,
                "negative_environment": row.get("negative_environment", ""),
                "negative_semantic_class": row.get("negative_semantic_class", ""),
                "anchor_id": row.get("anchor_id", ""),
                "true_partner_id": row.get("true_partner_id", ""),
                "candidate_partner_id": row.get("candidate_partner_id", ""),
            }
        )
    return resolved


@functools.cache
def panel_labels(root: Path, entry: PanelEntry) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Row ids and labels of one frozen panel, in file order.

    Cached because a full table reproduction touches the same panels hundreds
    of times; uses csv.reader instead of per-row dict allocation.
    """
    row_ids: list[str] = []
    labels: list[int] = []
    with open_text(root / entry.path) as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return (), ()
        row_id_idx = header.index("row_id")
        label_idx = header.index("interaction_label")
        for row in reader:
            row_ids.append(row[row_id_idx])
            labels.append(int(row[label_idx]))
    return tuple(row_ids), tuple(labels)


@functools.cache
def panel_positions(
    root: Path, entry: PanelEntry
) -> tuple[tuple[str, ...], dict[str, int], np.ndarray]:
    """Cached ``(row_ids, position_map, labels_array)`` for fast panel joins."""
    row_ids, labels = panel_labels(root, entry)
    position = {row_id: index for index, row_id in enumerate(row_ids)}
    return row_ids, position, np.asarray(labels, dtype=np.int64)


def verify_panel(root: Path, entry: PanelEntry) -> dict[str, Any]:
    path = root / entry.path
    if not path.exists():
        return {"path": entry.path, "status": "MISSING"}
    digest = sha256_file(path)
    row_ids, labels = panel_labels(root, entry)
    return {
        "path": entry.path,
        "status": "OK" if digest == entry.sha256 else "HASH_MISMATCH",
        "sha256": digest,
        "expected_sha256": entry.sha256,
        "rows": len(row_ids),
        "expected_rows": entry.rows,
        "n_positive": int(sum(labels)),
        "expected_n_positive": entry.n_positive,
        "duplicate_row_ids": len(row_ids) - len(set(row_ids)),
        "labels_binary": set(labels) <= {0, 1},
    }


def negative_environment_summary(root: Path, entry: PanelEntry) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in iter_panel_rows(root, entry):
        key = row.get("negative_environment") or "positive"
        counts[key] = counts.get(key, 0) + 1
    return counts


def check_environments_distinguishable(root: Path, entries: Sequence[PanelEntry]) -> dict[str, Any]:
    """The six negative environments must be separable by their own column."""
    seen: dict[str, set[str]] = {}
    for entry in entries:
        for row in iter_panel_rows(root, entry):
            if int(row["interaction_label"]) != 0:
                continue
            environment = row.get("negative_environment", "")
            seen.setdefault(environment, set()).add(entry.environment)
    return {
        "environments": sorted(seen),
        "expected": sorted(NEGATIVE_ENVIRONMENTS),
        "all_present": set(seen) >= set(NEGATIVE_ENVIRONMENTS),
        "per_environment_panels": {key: len(value) for key, value in sorted(seen.items())},
    }
