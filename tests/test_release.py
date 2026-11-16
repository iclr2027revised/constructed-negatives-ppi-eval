"""Tests for the released benchmark interface.

These run without the private research codebase and without the frozen panels:
they build a tiny synthetic panel and registry in a temporary directory.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Importing the released package must not write bytecode into the tree these
# tests assert is clean: a .pyc embeds the absolute path it was compiled at, and
# that path names the machine that built the release.
sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppi_shift_release import metrics, panels, predictions  # noqa: E402


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


LEFT = "MKTAYIAKQRQISFVKSHFSRQ"
RIGHT = "MSKPPPKPVKPGQVKVFRALYT"
OTHER = "MSTNPKPQRKTKRNTNRRPQDV"


@pytest.fixture()
def release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    (root / "data/manifests").mkdir(parents=True)
    (root / "data/sequence_registry").mkdir(parents=True)
    (root / "data/frozen_panels").mkdir(parents=True)

    sequences = {_sha(LEFT): LEFT, _sha(RIGHT): RIGHT, _sha(OTHER): OTHER}
    with gzip.open(root / panels.SEQUENCE_REGISTRY, "wt") as handle:
        for digest, sequence in sequences.items():
            handle.write(f">{digest}\n{sequence}\n")

    rows = [
        {
            "row_id": "positive:aaa",
            "interaction_label": "1",
            "anchor_sequence_sha256": _sha(LEFT),
            "true_partner_sequence_sha256": _sha(RIGHT),
            "candidate_sequence_sha256": "",
            "negative_environment": "",
            "negative_semantic_class": "curated_positive_reference",
            "anchor_id": "P1",
            "true_partner_id": "P2",
            "candidate_partner_id": "",
        },
        {
            "row_id": "negative:bbb",
            "interaction_label": "0",
            "anchor_sequence_sha256": _sha(LEFT),
            "true_partner_sequence_sha256": _sha(OTHER),
            "candidate_sequence_sha256": _sha(RIGHT),
            "negative_environment": "uniform_partner",
            "negative_semantic_class": "synthetic_unobserved_pair",
            "anchor_id": "P1",
            "true_partner_id": "P3",
            "candidate_partner_id": "P2",
        },
    ]
    panel_path = root / "data/frozen_panels/test__uniform_partner.csv"
    with panel_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "panels": [
            {
                "cohort": "E005",
                "split": "test",
                "environment": "uniform_partner",
                "path": "data/frozen_panels/test__uniform_partner.csv",
                "sha256": panels.sha256_file(panel_path),
                "rows": 2,
                "n_positive": 1,
                "n_negative": 1,
                "evidence_semantics": "synthetic_unobserved_pair",
            }
        ]
    }
    (root / panels.PANEL_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_sequence_registry_verifies(release: Path) -> None:
    report = panels.verify_sequence_registry(release)
    assert report["sequences"] == 3
    assert report["mismatched"] == 0


def test_panel_verifies(release: Path) -> None:
    entry = panels.load_manifest(release)[0]
    report = panels.verify_panel(release, entry)
    assert report["status"] == "OK"
    assert report["rows"] == 2
    assert report["duplicate_row_ids"] == 0
    assert report["labels_binary"] is True


def test_panel_resolves_to_real_sequences(release: Path) -> None:
    entry = panels.load_manifest(release)[0]
    sequences = panels.load_sequences(release)
    resolved = panels.resolve_panel(release, entry, sequences)
    assert len(resolved) == 2
    assert resolved[0]["left_sequence"] == LEFT
    assert resolved[0]["right_sequence"] == RIGHT
    assert resolved[0]["right_endpoint_role"] == "true_partner"
    # A negative scores the candidate, not the true partner it was derived from.
    assert resolved[1]["right_sequence"] == RIGHT
    assert resolved[1]["right_endpoint_role"] == "candidate_partner"
    assert resolved[1]["true_partner_id"] == "P3"


def test_missing_sequence_fails_loudly(release: Path) -> None:
    entry = panels.load_manifest(release)[0]
    sequences = panels.load_sequences(release)
    sequences.pop(_sha(RIGHT))
    with pytest.raises(KeyError, match="absent from the registry"):
        panels.resolve_panel(release, entry, sequences)


def test_environments_distinguishable(release: Path) -> None:
    entries = panels.load_manifest(release)
    report = panels.check_environments_distinguishable(release, entries)
    assert "uniform_partner" in report["environments"]


def test_metrics_match_sklearn() -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 400)
    scores = np.round(rng.random(400), 2)
    assert metrics.auroc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))
    assert metrics.auprc(labels, scores) == pytest.approx(
        average_precision_score(labels, scores)
    )


def test_score_conventions_preserve_cohort_specific_ties() -> None:
    labels = [0, 1]
    scores = [1.0 - 5e-8, 1.0]
    assert metrics.score_convention_for_cohort("E003_corrected_replay") == (
        metrics.CLIPPED_SCORE_CONVENTION
    )
    assert metrics.score_convention_for_cohort("E005_primary") == (
        metrics.RAW_SCORE_CONVENTION
    )
    assert metrics.auroc(labels, scores) == pytest.approx(1.0)
    assert metrics.auroc(
        labels, scores, score_convention=metrics.CLIPPED_SCORE_CONVENTION
    ) == pytest.approx(0.5)
    assert metrics.auprc(labels, scores) == pytest.approx(1.0)
    assert metrics.auprc(
        labels, scores, score_convention=metrics.CLIPPED_SCORE_CONVENTION
    ) == pytest.approx(0.5)


def test_prevalence_baseline_is_positive_rate() -> None:
    assert metrics.prevalence_baseline([1, 0, 0, 0]) == pytest.approx(0.25)


def test_single_class_rejected() -> None:
    with pytest.raises(ValueError, match="one class"):
        metrics.auroc([1, 1, 1], [0.1, 0.2, 0.3])


def test_predictions_reject_out_of_range(release: Path, tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("row_id,score\npositive:aaa,1.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        predictions.load_predictions(path)


def test_predictions_reject_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "dup.csv"
    path.write_text("row_id,score\npositive:aaa,0.4\npositive:aaa,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        predictions.load_predictions(path)


def test_predictions_reject_partial_coverage(release: Path, tmp_path: Path) -> None:
    entry = panels.load_manifest(release)[0]
    path = tmp_path / "partial.csv"
    path.write_text("row_id,score\npositive:aaa,0.9\n", encoding="utf-8")
    table = predictions.load_predictions(path)
    with pytest.raises(ValueError, match="have no prediction"):
        predictions.join_to_panel(release, entry, table)


def test_predictions_reject_unknown_rows(release: Path, tmp_path: Path) -> None:
    entry = panels.load_manifest(release)[0]
    path = tmp_path / "unknown.csv"
    path.write_text(
        "row_id,score\npositive:aaa,0.9\nnegative:bbb,0.1\nnegative:zzz,0.2\n",
        encoding="utf-8",
    )
    table = predictions.load_predictions(path)
    with pytest.raises(ValueError, match="absent from"):
        predictions.join_to_panel(release, entry, table)


def test_evaluate_produces_metrics(release: Path, tmp_path: Path) -> None:
    entry = panels.load_manifest(release)[0]
    path = tmp_path / "good.csv"
    path.write_text("row_id,score\npositive:aaa,0.9\nnegative:bbb,0.1\n", encoding="utf-8")
    table = predictions.load_predictions(path)
    result = predictions.evaluate(release, entry, table, bootstrap_replicates=50)
    assert result["n_rows"] == 2
    assert result["auroc"] == pytest.approx(1.0)
    assert result["prevalence"] == pytest.approx(0.5)


def test_paired_bootstrap_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 200)
    a = rng.random(200)
    b = rng.random(200)
    first = metrics.paired_row_bootstrap(labels, a, b, replicates=100, seed=7)
    second = metrics.paired_row_bootstrap(labels, a, b, replicates=100, seed=7)
    assert first == second
    assert first["lower"] <= first["point"] <= first["upper"]


def test_cohort_aliases_resolve_to_manifest_cohorts() -> None:
    """Every alias must reach a cohort the manifest actually carries."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import evaluate_predictions

    resolved = {
        evaluate_predictions.COHORT_ALIASES.get(name, name)
        for name in evaluate_predictions.COHORT_CHOICES
    }
    assert resolved == {
        "E003_corrected_replay",
        "E005_core",
        "E005_primary",
        "RBT_fully_random",
    }


def test_evaluate_predictions_reads_gzip(tmp_path: Path) -> None:
    """The scorer must read the container the release ships.

    Every pointwise file in the archive is ``.csv.gz``.  A reader who points
    ``evaluate_predictions.py`` at a shipped cell is doing the most obvious
    round-trip there is, so the reader has to accept gzip.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import evaluate_predictions

    target = tmp_path / "predictions.csv.gz"
    with gzip.open(target, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "score", "panel"])
        writer.writerow(["positive:aaa", "0.75", "uniform_partner"])
        writer.writerow(["negative:bbb", "0.25", "uniform_partner"])
        writer.writerow(["positive:ccc", "0.60", "degree_partner_matched"])

    grouped = evaluate_predictions._split_by_panel(target)
    assert set(grouped) == {"uniform_partner", "degree_partner_matched"}
    assert [row["score"] for row in grouped["uniform_partner"]] == ["0.75", "0.25"]
    assert [row["score"] for row in grouped["degree_partner_matched"]] == ["0.60"]


def test_broken_panel_resolves_the_scored_endpoint(tmp_path: Path) -> None:
    """A broken-pair positive scores the shifted partner, not the true partner.

    ``build_broken_pair_panel`` rewrites ``seq2`` for every row, positives
    included, and keeps the original partner in ``true_partner_*``.  A resolver
    that trusted ``true_partner_sequence_sha256`` would hand an external user a
    protein the model never scored.
    """
    root = tmp_path / "release"
    (root / "data/manifests").mkdir(parents=True)
    (root / "data/sequence_registry").mkdir(parents=True)
    (root / "data/frozen_panels").mkdir(parents=True)

    sequences = {_sha(LEFT): LEFT, _sha(RIGHT): RIGHT, _sha(OTHER): OTHER}
    with gzip.open(root / panels.SEQUENCE_REGISTRY, "wt") as handle:
        for digest, sequence in sequences.items():
            handle.write(f">{digest}\n{sequence}\n")

    rows = [
        {
            "row_id": "positive:aaa",
            "interaction_label": "1",
            "anchor_sequence_sha256": _sha(LEFT),
            # The original partner, which was not scored.
            "true_partner_sequence_sha256": _sha(RIGHT),
            "candidate_sequence_sha256": "",
            # The shifted partner, which was scored.
            "broken_candidate_sequence_sha256": _sha(OTHER),
            "negative_environment": "",
            "negative_semantic_class": "curated_positive_reference",
            "anchor_id": "P1",
            "true_partner_id": "P2",
            "candidate_partner_id": "",
        },
    ]
    panel_path = root / "data/frozen_panels/test__uniform_partner.csv"
    with panel_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "panels": [
            {
                "cohort": "E005_primary",
                "split": "test",
                "environment": "uniform_partner",
                "path": "data/frozen_panels/test__uniform_partner.csv",
                "sha256": panels.sha256_file(panel_path),
                "rows": 1,
                "n_positive": 1,
                "n_negative": 0,
                "evidence_semantics": "synthetic_unobserved_pair",
            }
        ]
    }
    (root / panels.PANEL_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

    entry = panels.load_manifest(root)[0]
    resolved = panels.resolve_panel(root, entry, panels.load_sequences(root))
    assert resolved[0]["right_sequence"] == OTHER
    assert resolved[0]["right_endpoint_role"] == "broken_candidate_partner"
    assert resolved[0]["right_sha256"] == _sha(OTHER)


# ---------------------------------------------------------------------------
# Metric-level reproduction layer
# ---------------------------------------------------------------------------

GOLD_SURFACES = ("tppni2025_scm_cl3_exact_balanced", "bernett2026_ilp_exact")
EXTERNAL_SURFACES = ("vidal__measured", "intact__measured")
E003_SOURCES = ("vidal", "intact")
MODELS = (
    "symmetric_frozen_esm2_pair_mlp",
    "tuna_corrected",
    "dscript_esm2",
    "rapppid",
)
MODEL_LABELS = {
    "symmetric_frozen_esm2_pair_mlp": "Frozen-ESM2 MLP",
    "tuna_corrected": "TUnA",
    "dscript_esm2": "D-SCRIPT",
    "rapppid": "RAPPPID",
}
SEEDS = (11, 22, 33)
GOLD_DELTA = {"tppni2025_scm_cl3_exact_balanced": 0.0100, "bernett2026_ilp_exact": -0.0200}
EXTERNAL_DELTA = {"vidal__measured": 0.0300, "intact__measured": 0.0400}


@pytest.fixture()
def metric_release(tmp_path: Path) -> Path:
    """A structurally faithful but tiny metric-level release.

    The fixture exercises the joins, the column contracts and the comparison
    logic.  It is not a miniature of the science: the numbers are chosen so that
    a wrong join or a wrong sign is visible.
    """

    import pandas as pd

    root = tmp_path / "metric_release"
    metric = root / "results/metric_level"
    metric.mkdir(parents=True)
    (root / "manifests").mkdir(parents=True)

    gold_rows = []
    for model in MODELS:
        for seed in SEEDS:
            for surface in GOLD_SURFACES:
                gold_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "test_environment": surface,
                        "delta_auroc": GOLD_DELTA[surface],
                        "delta_auprc": GOLD_DELTA[surface] / 2,
                    }
                )
    pd.DataFrame(gold_rows).to_csv(metric / "fully_random_gold_per_seed.csv", index=False)

    external_rows = []
    for model in MODELS:
        for seed in SEEDS:
            for surface in EXTERNAL_SURFACES:
                external_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "source": surface,
                        "delta_auroc": EXTERNAL_DELTA[surface],
                        "delta_auprc": EXTERNAL_DELTA[surface] / 2,
                    }
                )
    pd.DataFrame(external_rows).to_csv(
        metric / "fully_random_external_per_seed.csv", index=False
    )

    shortcut_rows = []
    for surface in GOLD_SURFACES + ("vidal__measured", "intact__measured"):
        shortcut_rows.append(
            {
                "metric": "auroc",
                "control": "length_composition",
                "test_environment": surface,
                "uniform_fit": 0.5,
                "fully_random_fit": 0.5 - 0.0077,
                "delta_fully_random_minus_uniform": -0.0077,
            }
        )
    pd.DataFrame(shortcut_rows).to_csv(metric / "shortcut_delta_symmetric.csv", index=False)

    agreement_rows = []
    for surface in GOLD_SURFACES + ("vidal__measured", "intact__measured"):
        agreement_rows.append(
            {
                "control": "length_composition",
                "environment": surface,
                "metric": "auroc",
                "records": 12,
                "defined": 12,
                "matches": 3,
                "fraction": 0.25,
            }
        )
    pd.DataFrame(agreement_rows).to_csv(
        metric / "sign_agreement_by_environment.csv", index=False
    )

    interval_rows = []
    for model in MODELS:
        for seed in SEEDS:
            crosses = model == "tuna_corrected" and seed == 11
            interval_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "seed": seed,
                    "environment": "vidal__measured",
                    "metric": "auroc",
                    "difference": 0.03,
                    "lower_paired_stratified_rows": 0.01,
                    "upper_paired_stratified_rows": 0.05,
                    "interval_crosses_zero_paired_stratified_rows": False,
                    "lower_left_endpoint_cluster": -0.0019 if crosses else 0.01,
                    "upper_left_endpoint_cluster": 0.0350 if crosses else 0.05,
                    "interval_crosses_zero_left_endpoint_cluster": crosses,
                }
            )
    pd.DataFrame(interval_rows).to_csv(
        metric / "statistical_per_seed_intervals.csv", index=False
    )

    e003_rows = []
    for model, base in (("dscript_esm2", 0.70), ("tuna_corrected", 0.75)):
        for seed in SEEDS:
            for source in E003_SOURCES:
                e003_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "source": source,
                        "auroc": base + (0.01 if source == "vidal" else 0.02),
                    }
                )
    pd.DataFrame(e003_rows).to_csv(metric / "e003_external_metric_summary.csv", index=False)

    e005_rows = []
    for model in MODELS:
        for seed in SEEDS:
            for surface in ("vidal__measured", "intact__measured"):
                e005_rows.append(
                    {
                        "model_id": model,
                        "surface_id": surface,
                        "seed": seed,
                        "auroc": 0.60 + (0.05 if surface == "vidal__measured" else 0.08),
                    }
                )
    pd.DataFrame(e005_rows).to_csv(metric / "e005_external_metrics.csv", index=False)

    expected = {
        "schema_version": 1,
        "fully_random_deltas": [
            {"model": MODEL_LABELS[model], "surface": surface, "value": value}
            for model in MODELS
            for surface, value in (
                ("vidal__measured", EXTERNAL_DELTA["vidal__measured"]),
                (
                    "tppni2025_scm_cl3_exact_balanced",
                    GOLD_DELTA["tppni2025_scm_cl3_exact_balanced"],
                ),
                ("bernett2026_ilp_exact", GOLD_DELTA["bernett2026_ilp_exact"]),
            )
        ],
        "direction_stability": [
            {"model": MODEL_LABELS[model], "stable_surfaces": 4} for model in MODELS
        ],
        "shortcut_deltas": [
            {"control": "length_composition", "surface": surface, "value": -0.0077}
            for surface in GOLD_SURFACES + ("vidal__measured", "intact__measured")
        ],
        "sign_agreement": [
            {
                "control": "length_composition",
                "surface": surface,
                "metric": "auroc",
                "matches": 3,
            }
            for surface in GOLD_SURFACES + ("vidal__measured", "intact__measured")
        ],
        "vidal_intervals": [
            {
                "model": MODEL_LABELS[model],
                "seed": seed,
                "cluster_crosses_zero": model == "tuna_corrected" and seed == 11,
            }
            for model in MODELS
            for seed in SEEDS
        ],
        "external_auroc_ranges": [
            {"cohort": "E003_corrected_replay", "source": "intact", "min": 0.72, "max": 0.77},
            {"cohort": "E003_corrected_replay", "source": "vidal", "min": 0.71, "max": 0.76},
            {"cohort": "E005_primary", "source": "intact", "min": 0.68, "max": 0.68},
            {"cohort": "E005_primary", "source": "vidal", "min": 0.65, "max": 0.65},
        ],
    }
    (root / "manifests/EXPECTED_VALUES.json").write_text(
        json.dumps(expected, indent=2), encoding="utf-8"
    )
    return root


def test_metric_level_verify_passes_on_consistent_tables(metric_release: Path) -> None:
    from ppi_shift_release import tables

    result = tables.verify(metric_release)
    assert result["status"] == "OK", [
        check for check in result["detail"] if check["status"] != "MATCH"
    ]
    assert result["checks"] == result["matched"]
    assert result["checks"] == 12 + 4 + 4 + 4 + 12 + 4


def test_metric_level_verify_flags_a_tampered_value(metric_release: Path) -> None:
    """A single edited cell must fail the check, not slip through."""

    import pandas as pd
    from ppi_shift_release import tables

    path = metric_release / "results/metric_level/fully_random_gold_per_seed.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "delta_auroc"] = 0.5
    frame.to_csv(path, index=False)

    result = tables.verify(metric_release)
    assert result["status"] == "FAILED"
    assert result["matched"] == result["checks"] - 1


def test_metric_level_verify_flags_a_flipped_interval(metric_release: Path) -> None:
    """The Vidal qualification is a claim about which interval crosses zero."""

    import pandas as pd
    from ppi_shift_release import tables

    path = metric_release / "results/metric_level/statistical_per_seed_intervals.csv"
    frame = pd.read_csv(path)
    mask = (frame["model_label"] == "TUnA") & (frame["seed"] == 11)
    frame.loc[mask, "interval_crosses_zero_left_endpoint_cluster"] = False
    frame.to_csv(path, index=False)

    result = tables.verify(metric_release)
    assert result["status"] == "FAILED"
    failed = [check for check in result["detail"] if check["status"] != "MATCH"]
    assert any("vidal_cluster_crosses_zero:TUnA:11" in check["claim"] for check in failed)


def test_unmapped_model_identifier_is_rejected(metric_release: Path) -> None:
    """A renamed model must raise, not silently drop out of the comparison."""

    import pandas as pd
    from ppi_shift_release import tables

    path = metric_release / "results/metric_level/fully_random_gold_per_seed.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "model"] = "some_other_model"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="unmapped model identifiers"):
        tables.fully_random_deltas(metric_release)


def test_missing_metric_table_names_the_file(tmp_path: Path) -> None:
    from ppi_shift_release import tables

    with pytest.raises(FileNotFoundError, match="fully_random_gold_per_seed.csv"):
        tables.load(tmp_path, "fully_random_gold_per_seed.csv")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


def test_pair_features_are_swap_invariant() -> None:
    """The symmetric baseline must not be able to read the storage order."""

    from ppi_shift_release.models import frozen_esm2_mlp

    rng = np.random.default_rng(0)
    vectors = {
        _sha(LEFT): rng.normal(size=frozen_esm2_mlp.EMBEDDING_DIM),
        _sha(RIGHT): rng.normal(size=frozen_esm2_mlp.EMBEDDING_DIM),
    }
    rows = [
        {
            "row_id": "positive:aaa",
            "interaction_label": "1",
            "anchor_sequence_sha256": _sha(LEFT),
            "true_partner_sequence_sha256": _sha(RIGHT),
            "candidate_sequence_sha256": "",
        }
    ]
    assert frozen_esm2_mlp.swap_invariance_error(rows, vectors) == pytest.approx(0.0, abs=1e-12)


def test_partner_hash_selects_the_scored_endpoint() -> None:
    """A negative scores the candidate; a positive scores the curated partner."""

    from ppi_shift_release.models import frozen_esm2_mlp

    positive = {
        "row_id": "positive:aaa",
        "interaction_label": "1",
        "true_partner_sequence_sha256": _sha(RIGHT),
        "candidate_sequence_sha256": "",
    }
    negative = {
        "row_id": "negative:bbb",
        "interaction_label": "0",
        "true_partner_sequence_sha256": _sha(OTHER),
        "candidate_sequence_sha256": _sha(RIGHT),
    }
    assert frozen_esm2_mlp.partner_hash(positive) == _sha(RIGHT)
    assert frozen_esm2_mlp.partner_hash(negative) == _sha(RIGHT)


def test_partner_hash_rejects_a_row_without_the_selected_endpoint() -> None:
    from ppi_shift_release.models import frozen_esm2_mlp

    with pytest.raises(ValueError, match="lacks the selected partner hash"):
        frozen_esm2_mlp.partner_hash(
            {
                "row_id": "negative:bbb",
                "interaction_label": "0",
                "candidate_sequence_sha256": "",
            }
        )


def test_pair_features_reject_wrong_embedding_width() -> None:
    from ppi_shift_release.models import frozen_esm2_mlp

    vectors = {_sha(LEFT): np.zeros(128), _sha(RIGHT): np.zeros(128)}
    rows = [
        {
            "row_id": "positive:aaa",
            "interaction_label": "1",
            "anchor_sequence_sha256": _sha(LEFT),
            "true_partner_sequence_sha256": _sha(RIGHT),
            "candidate_sequence_sha256": "",
        }
    ]
    with pytest.raises(ValueError, match="640-dimensional"):
        frozen_esm2_mlp.symmetric_pair_features(rows, vectors)


def test_rapppid_forward_shape() -> None:
    """The redistributed RAPPPID definition must build and run."""

    torch = pytest.importorskip("torch")
    from ppi_shift_release.models import rapppid

    model = rapppid.RapppidAWD(
        vocabulary_size=24, embedding_size=64, recurrent_layers=1,
        embedding_dropout=0.0, weight_dropout=0.0, class_head_dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        output = model(torch.ones(2, 12, dtype=torch.long), torch.ones(2, 12, dtype=torch.long))
    assert tuple(output.shape)[0] == 2


def test_dscript_esm2_forward_shape_and_provenance() -> None:
    """The executed D-SCRIPT contact CNN requires 640-d ESM2-150M tokens."""

    torch = pytest.importorskip("torch")
    from ppi_shift_release.models import dscript_esm2

    assert dscript_esm2.ESM2_EMBEDDING_DIM == 640
    assert dscript_esm2.EXPECTED_PROJECTION_WEIGHT_BYTES == 256000
    model = dscript_esm2.DScriptContactCNN()
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 10, 640), torch.zeros(2, 12, 640))
    assert tuple(out.shape) == (2,)
    with pytest.raises(ValueError, match="640"):
        dscript_esm2.DScriptContactCNN(input_dim=1280)



# ---------------------------------------------------------------------------
# Release-level contracts
# ---------------------------------------------------------------------------

RELEASE_ROOT = Path(__file__).resolve().parents[1]


def test_coverage_matrix_declares_every_level() -> None:
    """The coverage matrix is the package's claim about what is reproducible."""

    path = RELEASE_ROOT / "manifests/COVERAGE_MATRIX.csv"
    if not path.exists():
        pytest.skip("coverage matrix is written by the release builder")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    levels = {row["level"] for row in rows}
    assert {"1_pointwise", "1_metric", "2_inference", "3_training"} <= levels
    for row in rows:
        assert row["status"], row
        assert row["reproduction"], row
        if row["status"].startswith("BLOCKED"):
            assert row["limitation"], row


def test_integrity_manifest_covers_every_distributed_file() -> None:
    """Every shipped file must be listed, and every listed file must exist."""

    path = RELEASE_ROOT / "manifests/INTEGRITY.json"
    if not path.exists():
        pytest.skip("integrity manifest is written by the release builder")

    manifest = json.loads(path.read_text(encoding="utf-8"))
    listed = set(manifest["files"])
    on_disk = {
        candidate.relative_to(RELEASE_ROOT).as_posix()
        for candidate in RELEASE_ROOT.rglob("*")
        if candidate.is_file()
    }
    # Outputs written by running the package are not distributed inputs, and a
    # virtualenv created inside the checkout is not part of the release.
    generated = (
        ".git/",
        ".pytest_cache/",
        ".venv/",
        "venv/",
        "env/",
        "build/",
        "dist/",
        "results/figures_regenerated/",
        "results/tables/reproduction_report",
        "results/tables/validation_report",
        "manifests/validation_report",
    )
    on_disk = {
        name
        for name in on_disk
        if not name.startswith(generated)
        and "__pycache__" not in name
        and ".egg-info/" not in name
        and not name.endswith(".pyc")
    }
    assert listed <= on_disk
    assert on_disk - listed <= {"manifests/INTEGRITY.json"}


def test_release_names_no_author_machine_or_private_commit(tmp_path: Path) -> None:
    """No distributed byte may name an author, a machine or a private commit.

    ``scripts/validate_release.py`` is the scanner that knows about Git commit
    identifiers.  It is run here, through its real command line, so a release
    that reintroduces a private SHA fails the shipped test suite.
    """

    scanner = RELEASE_ROOT / "scripts/validate_release.py"
    if not scanner.exists():
        pytest.skip("release scanner is not part of this checkout")

    report = tmp_path / "validation_report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(scanner),
            "--root",
            str(RELEASE_ROOT),
            "--skip-integrity",
            "--strict-artifacts",
            "--output",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert report.exists(), completed.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    findings = payload["anonymity_findings"]
    assert not findings, findings[:10]
    assert payload["status"] == "OK", payload["problems"]


def test_anonymity_scanner_detects_synthetic_path_fixture(tmp_path: Path) -> None:
    """The generic scanner detects a synthetic path assembled at runtime."""

    scanner = RELEASE_ROOT / "scripts/validate_release.py"
    fixture = (
        "C:"
        + chr(92)
        + "Users"
        + chr(92)
        + "example-reviewer"
        + chr(92)
        + "private"
        + chr(92)
        + "sample.txt"
    )
    (tmp_path / "fixture.txt").write_text(fixture, encoding="utf-8")
    report = tmp_path / "scan-report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(scanner),
            "--root",
            str(tmp_path),
            "--skip-integrity",
            "--output",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert report.exists(), completed.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert any(
        finding["kind"] == "windows_user_path"
        for finding in payload["anonymity_findings"]
    )


def test_anonymity_scanner_rejects_unattributed_repository_url(tmp_path: Path) -> None:
    """Known third-party links are allow-listed; arbitrary repo links still fail."""

    scanner = RELEASE_ROOT / "scripts/validate_release.py"
    fixture = bytes(
        [
            103, 105, 116, 104, 117, 98, 46, 99, 111, 109, 47, 101, 120, 97, 109,
            112, 108, 101, 45, 114, 101, 118, 105, 101, 119, 101, 114, 47, 112, 114,
            111, 106, 101, 99, 116,
        ]
    ).decode("ascii")
    (tmp_path / "fixture.txt").write_text(fixture, encoding="utf-8")
    allow = tmp_path / "allow.json"
    allow.write_text("[]\n", encoding="utf-8")
    report = tmp_path / "scan-report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(scanner),
            "--root",
            str(tmp_path),
            "--skip-integrity",
            "--allow",
            str(allow),
            "--output",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert any(
        finding["kind"] == "repository_url"
        for finding in payload["anonymity_findings"]
    )


def test_expected_values_manifest_is_not_empty() -> None:
    path = RELEASE_ROOT / "manifests/EXPECTED_VALUES.json"
    if not path.exists():
        pytest.skip("expected values are written by the release builder")

    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "fully_random_deltas",
        "direction_stability",
        "shortcut_deltas",
        "sign_agreement",
        "vidal_intervals",
        "external_auroc_ranges",
    ):
        assert payload[key], key


def test_anonymity_scanner_rejects_institutional_gitlab_url(tmp_path: Path) -> None:
    """Non-.com institutional GitLab/Bitbucket/Codeberg URLs must also be caught."""

    scanner = RELEASE_ROOT / "scripts/validate_release.py"
    fixture = bytes(
        [
            104, 116, 116, 112, 115, 58, 47, 47, 103, 105, 116, 108, 97, 98, 46,
            101, 120, 97, 109, 112, 108, 101, 45, 117, 110, 105, 118, 46, 102, 114,
            47, 108, 97, 98, 117, 115, 101, 114, 47, 112, 114, 111, 106, 101, 99, 116,
        ]
    ).decode("ascii")
    (tmp_path / "fixture.txt").write_text(fixture, encoding="utf-8")
    allow = tmp_path / "allow.json"
    allow.write_text("[]\n", encoding="utf-8")
    report = tmp_path / "scan-gitlab.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(scanner),
            "--root",
            str(tmp_path),
            "--skip-integrity",
            "--allow",
            str(allow),
            "--output",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert any(
        finding["kind"] == "repository_url"
        for finding in payload["anonymity_findings"]
    )
