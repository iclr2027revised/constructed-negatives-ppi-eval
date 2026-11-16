"""Metric-level reproduction from the released per-seed metric tables.

The pointwise layer in ``results/pointwise`` covers 864 evaluation cells: the
complete E005 primary cohort and complete E003 corrected-replay cohort. Three
other result families remain **not** distributed at pointwise level, because
their prediction files were produced on the authors' cluster and are not part of
the redistributable artifact set:

* the two-endpoint Fully Random training arm,
* the Vidal/Y2H and IntAct measured-source evaluations,
* E003 original cells outside the corrected replay.

For those surfaces the release ships the per-seed metric tables that the
manuscript numbers were computed from.  This module recomputes every reported
quantity from those tables and checks it against
``manifests/EXPECTED_VALUES.json``.  That is a genuine reproduction of the
reported numbers, and it is explicitly *not* a reproduction from predictions.
The distinction is recorded per claim in ``manifests/COVERAGE_MATRIX.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

METRIC_LEVEL = "results/metric_level"
EXPECTED = "manifests/EXPECTED_VALUES.json"

# Model identifiers as they appear in the released tables, mapped to the labels
# the manuscript uses.  Keeping the mapping explicit stops a silent rename from
# turning a missing model into a passing check.
MODEL_LABELS = {
    "symmetric_frozen_esm2_pair_mlp": "Frozen-ESM2 MLP",
    "tuna_corrected": "TUnA",
    "dscript_esm2": "D-SCRIPT",
    "rapppid": "RAPPPID",
}

GOLD_ENVIRONMENTS = (
    "uniform_partner",
    "degree_partner_matched",
    "sequence_nuisance_matched",
    "esm_partner_matched",
    "bernett2026_ilp_exact",
    "tppni2025_scm_cl3_exact_balanced",
)

MEASURED_SOURCES = ("vidal__measured", "intact__measured")


def load(root: Path, name: str) -> pd.DataFrame:
    """Load one released metric-level table."""

    path = root / METRIC_LEVEL / name
    if not path.exists():
        raise FileNotFoundError(f"missing metric-level table: {path}")
    return pd.read_csv(path)


def fully_random_deltas(root: Path) -> pd.DataFrame:
    """Per-model mean Fully Random minus Uniform AUROC, per evaluation surface.

    The mean is taken over the three fixed training seeds.  It is a descriptive
    average of dependent records, not a confidence interval.
    """

    gold = load(root, "fully_random_gold_per_seed.csv")
    external = load(root, "fully_random_external_per_seed.csv")

    gold = gold.rename(columns={"test_environment": "surface"})
    external = external.rename(columns={"source": "surface"})
    columns = ["model", "seed", "surface", "delta_auroc", "delta_auprc"]
    combined = pd.concat([gold[columns], external[columns]], ignore_index=True)
    combined["model_label"] = combined["model"].map(MODEL_LABELS)
    if combined["model_label"].isna().any():
        unknown = sorted(combined.loc[combined["model_label"].isna(), "model"].unique())
        raise ValueError(f"unmapped model identifiers: {unknown}")

    grouped = (
        combined.groupby(["model_label", "surface"], as_index=False)
        .agg(
            mean_delta_auroc=("delta_auroc", "mean"),
            mean_delta_auprc=("delta_auprc", "mean"),
            seeds=("seed", "nunique"),
        )
        .sort_values(["model_label", "surface"])
    )
    return grouped


def direction_stability(root: Path) -> pd.DataFrame:
    """Count surfaces on which all three seeds share the mean's sign."""

    gold = load(root, "fully_random_gold_per_seed.csv").rename(
        columns={"test_environment": "surface"}
    )
    external = load(root, "fully_random_external_per_seed.csv").rename(
        columns={"source": "surface"}
    )
    columns = ["model", "seed", "surface", "delta_auroc"]
    combined = pd.concat([gold[columns], external[columns]], ignore_index=True)
    combined["model_label"] = combined["model"].map(MODEL_LABELS)

    rows: list[dict[str, Any]] = []
    for (model, surface), part in combined.groupby(["model_label", "surface"]):
        mean = float(part["delta_auroc"].mean())
        signs = set((part["delta_auroc"] > 0).tolist())
        rows.append(
            {
                "model_label": model,
                "surface": surface,
                "mean_delta_auroc": mean,
                "seeds": int(part["seed"].nunique()),
                "stable": len(signs) == 1,
            }
        )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("model_label", as_index=False)
        .agg(stable_surfaces=("stable", "sum"), surfaces=("stable", "size"))
        .sort_values("model_label")
    )
    return summary


def shortcut_deltas(root: Path) -> pd.DataFrame:
    """Symmetric endpoint-control deltas, one row per control and surface."""

    frame = load(root, "shortcut_delta_symmetric.csv")
    return frame.rename(
        columns={
            "control": "control_id",
            "test_environment": "surface",
            "delta_fully_random_minus_uniform": "delta",
        }
    )


def sign_agreement(root: Path) -> pd.DataFrame:
    """Agreement between the primary control's sign and the neural shift's sign."""

    frame = load(root, "sign_agreement_by_environment.csv")
    return frame.rename(columns={"environment": "surface"})


def vidal_seed_intervals(root: Path) -> pd.DataFrame:
    """Per-seed Vidal AUROC intervals under both sensitivity constructions."""

    frame = load(root, "statistical_per_seed_intervals.csv")
    subset = frame[(frame["environment"] == "vidal__measured") & (frame["metric"] == "auroc")]
    return subset[
        [
            "model_label",
            "seed",
            "difference",
            "lower_paired_stratified_rows",
            "upper_paired_stratified_rows",
            "interval_crosses_zero_paired_stratified_rows",
            "lower_left_endpoint_cluster",
            "upper_left_endpoint_cluster",
            "interval_crosses_zero_left_endpoint_cluster",
        ]
    ].sort_values(["model_label", "seed"])


def external_auroc_ranges(root: Path) -> pd.DataFrame:
    """Per-cohort AUROC range across the six training constructions.

    E003 is read from the released E003 external summary; E005 from the verified
    E005 external table.  Both are seed-level records averaged over the three
    seeds before the range is taken, matching the manuscript's wording.
    """

    rows: list[dict[str, Any]] = []

    e003 = load(root, "e003_external_metric_summary.csv")
    e003 = e003[e003["source"].isin(["vidal", "intact"])]
    e003_mean = e003.groupby(["model", "source"], as_index=False)["auroc"].mean()
    for source, part in e003_mean.groupby("source"):
        rows.append(
            {
                "cohort": "E003_corrected_replay",
                "source": source,
                "min_auroc": float(part["auroc"].min()),
                "max_auroc": float(part["auroc"].max()),
            }
        )

    e005 = load(root, "e005_external_metrics.csv")
    e005 = e005[e005["surface_id"].isin(MEASURED_SOURCES)]
    e005_mean = e005.groupby(["model_id", "surface_id"], as_index=False)["auroc"].mean()
    for surface, part in e005_mean.groupby("surface_id"):
        rows.append(
            {
                "cohort": "E005_primary",
                "source": surface.replace("__measured", ""),
                "min_auroc": float(part["auroc"].min()),
                "max_auroc": float(part["auroc"].max()),
            }
        )

    return pd.DataFrame(rows).sort_values(["cohort", "source"])


def _missing(claim: str, expected: Any) -> dict[str, Any]:
    """A claim whose source row is absent from the released tables.

    The ``claim`` key is always present so a caller can report which claim was
    absent rather than crashing on a missing field.
    """

    return {"claim": claim, "expected": expected, "observed": None, "status": "MISSING"}


def _close(observed: float, expected: float, tolerance: float) -> bool:
    return abs(observed - expected) <= tolerance


def verify(root: Path, *, tolerance: float = 5e-4) -> dict[str, Any]:
    """Recompute every metric-level claim and compare it with the manifest.

    ``tolerance`` is 5e-4 because the manuscript quotes four decimal places and
    the released tables carry full precision; a tighter bound would fail on
    rounding rather than on substance.
    """

    expected = json.loads((root / EXPECTED).read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    deltas = fully_random_deltas(root)
    index = {(row.model_label, row.surface): row for row in deltas.itertuples()}
    for claim in expected["fully_random_deltas"]:
        key = (claim["model"], claim["surface"])
        row = index.get(key)
        if row is None:
            checks.append(
                _missing(
                    f"fully_random_delta_auroc:{claim['model']}:{claim['surface']}",
                    claim["value"],
                )
            )
            continue
        observed = float(row.mean_delta_auroc)
        checks.append(
            {
                "claim": f"fully_random_delta_auroc:{claim['model']}:{claim['surface']}",
                "expected": claim["value"],
                "observed": observed,
                "status": "MATCH" if _close(observed, claim["value"], tolerance) else "MISMATCH",
            }
        )

    stability = direction_stability(root)
    stable_index = {row.model_label: row for row in stability.itertuples()}
    for claim in expected["direction_stability"]:
        row = stable_index.get(claim["model"])
        if row is None:
            checks.append(
                _missing(
                    f"direction_stability:{claim['model']}", claim["stable_surfaces"]
                )
            )
            continue
        observed = int(row.stable_surfaces)
        checks.append(
            {
                "claim": f"direction_stability:{claim['model']}",
                "expected": claim["stable_surfaces"],
                "observed": observed,
                "status": "MATCH" if observed == claim["stable_surfaces"] else "MISMATCH",
            }
        )

    shortcuts = shortcut_deltas(root)
    shortcut_index = {
        (row.control_id, row.surface): float(row.delta) for row in shortcuts.itertuples()
    }
    for claim in expected["shortcut_deltas"]:
        key = (claim["control"], claim["surface"])
        observed = shortcut_index.get(key)
        if observed is None:
            checks.append(
                _missing(
                    f"shortcut_delta:{claim['control']}:{claim['surface']}", claim["value"]
                )
            )
            continue
        checks.append(
            {
                "claim": f"shortcut_delta:{claim['control']}:{claim['surface']}",
                "expected": claim["value"],
                "observed": observed,
                "status": "MATCH" if _close(observed, claim["value"], tolerance) else "MISMATCH",
            }
        )

    agreement = sign_agreement(root)
    agreement_index = {
        (row.control, row.surface, row.metric): int(row.matches) for row in agreement.itertuples()
    }
    for claim in expected["sign_agreement"]:
        key = (claim["control"], claim["surface"], claim["metric"])
        observed = agreement_index.get(key)
        if observed is None:
            checks.append(
                _missing(
                    f"sign_agreement:{claim['control']}:{claim['surface']}:{claim['metric']}",
                    claim["matches"],
                )
            )
            continue
        checks.append(
            {
                "claim": f"sign_agreement:{claim['control']}:{claim['surface']}:{claim['metric']}",
                "expected": claim["matches"],
                "observed": observed,
                "status": "MATCH" if observed == claim["matches"] else "MISMATCH",
            }
        )

    intervals = vidal_seed_intervals(root)
    interval_index = {(row.model_label, int(row.seed)): row for row in intervals.itertuples()}
    for claim in expected["vidal_intervals"]:
        key = (claim["model"], claim["seed"])
        row = interval_index.get(key)
        if row is None:
            checks.append(
                _missing(
                    f"vidal_cluster_crosses_zero:{claim['model']}:{claim['seed']}",
                    claim["cluster_crosses_zero"],
                )
            )
            continue
        observed = bool(row.interval_crosses_zero_left_endpoint_cluster)
        checks.append(
            {
                "claim": f"vidal_cluster_crosses_zero:{claim['model']}:{claim['seed']}",
                "expected": claim["cluster_crosses_zero"],
                "observed": observed,
                "status": "MATCH" if observed == claim["cluster_crosses_zero"] else "MISMATCH",
            }
        )

    ranges = external_auroc_ranges(root)
    range_index = {(row.cohort, row.source): row for row in ranges.itertuples()}
    for claim in expected["external_auroc_ranges"]:
        key = (claim["cohort"], claim["source"])
        row = range_index.get(key)
        if row is None:
            checks.append(
                _missing(
                    f"external_auroc_range:{claim['cohort']}:{claim['source']}",
                    [claim["min"], claim["max"]],
                )
            )
            continue
        low_ok = _close(float(row.min_auroc), claim["min"], tolerance)
        high_ok = _close(float(row.max_auroc), claim["max"], tolerance)
        checks.append(
            {
                "claim": f"external_auroc_range:{claim['cohort']}:{claim['source']}",
                "expected": [claim["min"], claim["max"]],
                "observed": [float(row.min_auroc), float(row.max_auroc)],
                "status": "MATCH" if low_ok and high_ok else "MISMATCH",
            }
        )

    matched = sum(1 for check in checks if check["status"] == "MATCH")
    return {
        "status": "OK" if matched == len(checks) else "FAILED",
        "checks": len(checks),
        "matched": matched,
        "tolerance": tolerance,
        "detail": checks,
    }
