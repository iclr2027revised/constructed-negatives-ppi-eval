"""Independent two-endpoint random-pair intervention; frozen E005 is read-only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ppi_shift.artifacts import sha256_path

EXPERIMENT = "RBT_RANDOM_PAIR_1024"
ENVIRONMENT = "fully_random_pair"
SEEDS = (11, 22, 33)
MODELS = ("symmetric_frozen_esm2_pair_mlp", "tuna_corrected", "dscript_esm2", "rapppid")
BACKEND_COLUMNS = (
    "row_id",
    "left_id",
    "right_id",
    "left_sequence",
    "right_sequence",
    "left_sequence_sha256",
    "right_sequence_sha256",
    "label",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checked_path(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    if sha256_path(path) != spec["sha256"]:
        raise ValueError(f"Input SHA256 mismatch: {path}")
    return path


def file_spec(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_path(path)}


def pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def pair_rank(i: int, j: int, n: int) -> int:
    if not 0 <= i < j < n:
        raise ValueError("Pair must belong to the strict upper triangle")
    return i * (2 * n - i - 1) // 2 + j - i - 1


def unrank_pair(rank: int, n: int) -> tuple[int, int]:
    if not 0 <= rank < n * (n - 1) // 2:
        raise ValueError("Pair rank outside upper triangle")
    i = (2 * n - 1 - math.isqrt((2 * n - 1) ** 2 - 8 * rank)) // 2
    while i * (2 * n - i - 1) // 2 > rank:
        i -= 1
    return i, i + 1 + rank - i * (2 * n - i - 1) // 2


def sample_pairs(
    proteins: dict[str, tuple[str, str]],
    forbidden_ids: set[tuple[str, str]],
    forbidden_sequences: set[tuple[str, str]],
    count: int,
    seed: int,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """PCG64 uniform sample without replacement of legal unordered ID pairs.

    Enumerating ranks, rather than anchoring on positive rows, makes the legal
    population and its exact inclusion probability explicit. Sequence aliases
    of permitted reference positives are excluded before sampling.
    """
    ids = sorted(proteins)
    n = len(ids)
    positions = {p: i for i, p in enumerate(ids)}
    by_sequence: dict[str, list[int]] = {}
    for p in ids:
        by_sequence.setdefault(proteins[p][1], []).append(positions[p])
    banned = set()
    for a, b in forbidden_ids:
        if a in positions and b in positions and a != b:
            banned.add(pair_rank(*sorted((positions[a], positions[b])), n))
    for a, b in forbidden_sequences:
        for i in by_sequence.get(a, []):
            for j in by_sequence.get(b, []):
                if i != j:
                    banned.add(pair_rank(*sorted((i, j)), n))
    total = n * (n - 1) // 2
    legal = np.setdiff1d(
        np.arange(total, dtype=np.int64),
        np.asarray(sorted(banned), dtype=np.int64),
        assume_unique=True,
    )
    if count < 1 or count > len(legal):
        raise ValueError(f"Legal pair shortage: requested={count}, legal={len(legal)}")
    ranks = np.random.Generator(np.random.PCG64(seed)).choice(legal, size=count, replace=False)
    pairs = [(ids[i], ids[j]) for i, j in (unrank_pair(int(k), n) for k in ranks)]
    return pairs, {
        "algorithm": "numpy.PCG64.choice(strict_upper_triangle_legal_ranks, replace=False)",
        "seed": seed,
        "proteins": n,
        "upper_triangle_pairs": total,
        "excluded_pairs": len(banned),
        "legal_pairs": len(legal),
        "marginal_pair_inclusion_probability": count / len(legal),
    }


def proteins_from_positives(rows: list[dict[str, str]]) -> dict[str, tuple[str, str]]:
    proteins: dict[str, tuple[str, str]] = {}
    for row in rows:
        if row["interaction_label"] != "1":
            continue
        for identifier, sequence, digest in (
            (row["anchor_id"], row["seq1"], row["anchor_sequence_sha256"]),
            (row["true_partner_id"], row["seq2"], row["true_partner_sequence_sha256"]),
        ):
            if not sequence or len(sequence) > 1024:
                raise ValueError("Sequence outside frozen E005 length contract")
            if hashlib.sha256(sequence.encode()).hexdigest() != digest:
                raise ValueError("Sequence SHA256 mismatch")
            if proteins.setdefault(identifier, (sequence, digest)) != (sequence, digest):
                raise ValueError(f"Ambiguous protein sequence: {identifier}")
    return proteins


def load_registry(
    spec: dict[str, Any],
    stage: str,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    if stage not in ("train", "validation") or spec["name"] != f"{stage}_exclusion_registry":
        raise ValueError("Forbidden registry role")
    allowed = {"gold_train"} if stage == "train" else {"gold_train", "gold_validation"}
    ids, sequences = set(), set()
    with checked_path(spec).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["source_dataset"] not in allowed:
                raise ValueError("Forbidden reference labels in training/validation registry")
            ids.add(pair_key(*row["identifier_pair_key"].split("||")))
            sequences.add(pair_key(*row["sequence_pair_key"].split("||")))
    return ids, sequences


def make_rows(
    positives: list[dict[str, str]],
    pairs: list[tuple[str, str]],
    proteins: dict[str, tuple[str, str]],
    stage: str,
    seed: int = 20261106,
) -> list[dict[str, str]]:
    if len(positives) != len(pairs) or any(p["interaction_label"] != "1" for p in positives):
        raise ValueError("Require exactly one random negative per frozen positive")
    result = []
    for index, (positive, (a, b)) in enumerate(zip(positives, pairs, strict=True)):
        # Preserve positive source rows byte-for-field. Parent IDs on negatives
        # are bookkeeping only: neither endpoint is sampled from that positive.
        negative = dict(positive)
        negative.update(
            {
                "row_id": f"rbt:{stage}:negative:{index:08d}",
                "interaction_label": "0",
                "anchor_id": a,
                "candidate_partner_id": b,
                "anchor_sequence_sha256": proteins[a][1],
                "candidate_sequence_sha256": proteins[b][1],
                "seq1": proteins[a][0],
                "seq2": proteins[b][0],
                "negative_environment": ENVIRONMENT,
                "negative_semantic_class": "synthetic_unobserved_pair",
                "name1": a,
                "name2": b,
                "sampler_version": "ppi_shift.rebuttal_random_pair.v1",
                "generation_seed": str(seed),
                "anchor_orientation_rule": "lexicographic_unordered_pair",
                "candidate_universe_id": f"{EXPERIMENT}:{stage}",
                "eligible_candidate_count": "",
                "candidate_rank": "",
                "match_distance": "",
                "match_diagnostics_json": "{}",
                "fallback_status": "none",
                "rejection_reason": "",
            }
        )
        result.extend((dict(positive), negative))
    return result


def audit_rows(
    rows: list[dict[str, str]],
    positives: list[dict[str, str]],
    proteins: dict[str, tuple[str, str]],
    forbidden_ids: set[tuple[str, str]],
    forbidden_sequences: set[tuple[str, str]],
) -> dict[str, Any]:
    if [r for r in rows if r["interaction_label"] == "1"] != positives:
        raise ValueError("Frozen positive backbone changed")
    negatives = [r for r in rows if r["interaction_label"] == "0"]
    if len(rows) != 2 * len(positives) or len(negatives) != len(positives):
        raise ValueError("Class balance changed")
    if len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate row identities")
    pairs = set()
    seqpairs = set()
    for r in negatives:
        a, b = r["anchor_id"], r["candidate_partner_id"]
        pair = pair_key(a, b)
        spair = pair_key(r["anchor_sequence_sha256"], r["candidate_sequence_sha256"])
        if a == b or pair in pairs or pair in forbidden_ids or spair in forbidden_sequences:
            raise ValueError("Illegal, duplicate or reference-positive random pair")
        if a not in proteins or b not in proteins:
            raise ValueError("Stage-ineligible endpoint")
        if (r["seq1"], r["anchor_sequence_sha256"]) != proteins[a]:
            raise ValueError("Left sequence/hash differs")
        if (r["seq2"], r["candidate_sequence_sha256"]) != proteins[b]:
            raise ValueError("Right sequence/hash differs")
        pairs.add(pair)
        seqpairs.add(spair)
    return {
        "positive_rows": len(positives),
        "negative_rows": len(negatives),
        "unique_negative_pairs": len(pairs),
        "unique_negative_sequence_pairs": len(seqpairs),
        "known_positive_collisions": 0,
        "self_pairs": 0,
        "reversed_duplicates": 0,
        "negative_semantics": "synthetic_unobserved_pair",
        "reference_scope": "stage-visible Gold ID and exact-sequence registry only",
        "biological_false_negative_rate": "unknown",
    }


def assert_stage_isolation(stages: dict[str, dict[str, tuple[str, str]]]) -> None:
    for i, (a, pa) in enumerate(stages.items()):
        for b, pb in list(stages.items())[i + 1 :]:
            if set(pa) & set(pb) or {v[1] for v in pa.values()} & {v[1] for v in pb.values()}:
                raise ValueError(f"Protein or exact-sequence leakage: {a}/{b}")


def backend_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    from ppi_shift.e003_runtime import d007_rows

    return d007_rows(rows)


def write_rows(path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    spec = {
        **file_spec(path),
        "input_sha256": sha256_path(path),
        "rows": len(rows),
    }
    if "row_id" in rows[0]:
        spec["row_ids_sha256"] = hashlib.sha256(
            "\n".join(r["row_id"] for r in rows).encode()
        ).hexdigest()
    return spec


def endpoint_diagnostics(rows: list[dict[str, str]], universe: list[str]) -> list[dict[str, Any]]:
    result = []
    for label in ("1", "0"):
        selected = [r for r in rows if r["interaction_label"] == label]
        degree = Counter()
        first, second = Counter(), Counter()
        for r in selected:
            a = r["anchor_id"]
            b = r["true_partner_id"] if label == "1" else r["candidate_partner_id"]
            degree.update((a, b))
            first[a] += 1
            second[b] += 1
        for p in universe:
            result.append(
                {
                    "protein_id": p,
                    "label": label,
                    "degree": degree[p],
                    "first_endpoint": first[p],
                    "second_endpoint": second[p],
                }
            )
    return result
