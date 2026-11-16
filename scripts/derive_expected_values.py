"""Derive the expected-value manifest from the released tables.

Run once at build time. The output is committed as
``manifests/EXPECTED_VALUES.json`` and is what ``reproduce.py`` checks against.
Deriving it from the tables rather than typing it from the manuscript means the
manifest cannot silently disagree with the data it describes; the manuscript
cross-check is a separate step in the release verification.
"""

from __future__ import annotations

import json
import sys

sys.dont_write_bytecode = True
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1]).resolve()
METRIC = ROOT / "results/metric_level"

MODEL_LABELS = {
    "symmetric_frozen_esm2_pair_mlp": "Frozen-ESM2 MLP",
    "tuna_corrected": "TUnA",
    "dscript_esm2": "D-SCRIPT",
    "rapppid": "RAPPPID",
}

REPORTED_SURFACES = (
    "vidal__measured",
    "tppni2025_scm_cl3_exact_balanced",
    "bernett2026_ilp_exact",
)


def main() -> int:
    gold = pd.read_csv(METRIC / "fully_random_gold_per_seed.csv").rename(
        columns={"test_environment": "surface"}
    )
    external = pd.read_csv(METRIC / "fully_random_external_per_seed.csv").rename(
        columns={"source": "surface"}
    )
    columns = ["model", "seed", "surface", "delta_auroc"]
    combined = pd.concat([gold[columns], external[columns]], ignore_index=True)
    combined["model_label"] = combined["model"].map(MODEL_LABELS)

    deltas = []
    for (model, surface), part in combined.groupby(["model_label", "surface"]):
        if surface not in REPORTED_SURFACES:
            continue
        deltas.append(
            {
                "model": model,
                "surface": surface,
                "value": round(float(part["delta_auroc"].mean()), 4),
            }
        )

    stability = []
    for model, part in combined.groupby("model_label"):
        stable = 0
        for _, surface_part in part.groupby("surface"):
            if len(set((surface_part["delta_auroc"] > 0).tolist())) == 1:
                stable += 1
        stability.append({"model": model, "stable_surfaces": stable})

    shortcuts = pd.read_csv(METRIC / "shortcut_delta_symmetric.csv")
    shortcuts = shortcuts[
        (shortcuts["control"] == "length_composition") & (shortcuts["metric"] == "auroc")
    ]
    shortcut_rows = [
        {
            "control": "length_composition",
            "surface": row.test_environment,
            "value": round(float(row.delta_fully_random_minus_uniform), 4),
        }
        for row in shortcuts.itertuples()
    ]

    agreement = pd.read_csv(METRIC / "sign_agreement_by_environment.csv")
    agreement = agreement[agreement["control"] == "length_composition"]
    agreement_rows = [
        {
            "control": "length_composition",
            "surface": row.environment,
            "metric": row.metric,
            "matches": int(row.matches),
        }
        for row in agreement.itertuples()
    ]

    intervals = pd.read_csv(METRIC / "statistical_per_seed_intervals.csv")
    intervals = intervals[
        (intervals["environment"] == "vidal__measured") & (intervals["metric"] == "auroc")
    ]
    interval_rows = [
        {
            "model": row.model_label,
            "seed": int(row.seed),
            "cluster_crosses_zero": bool(row.interval_crosses_zero_left_endpoint_cluster),
        }
        for row in intervals.itertuples()
    ]

    e003 = pd.read_csv(METRIC / "e003_external_metric_summary.csv")
    e003 = e003[e003["source"].isin(["vidal", "intact"])]
    e003_mean = e003.groupby(["model", "source"], as_index=False)["auroc"].mean()
    e005 = pd.read_csv(METRIC / "e005_external_metrics.csv")
    e005 = e005[e005["surface_id"].isin(["vidal__measured", "intact__measured"])]
    e005_mean = e005.groupby(["model_id", "surface_id"], as_index=False)["auroc"].mean()

    ranges = []
    for source, part in e003_mean.groupby("source"):
        ranges.append(
            {
                "cohort": "E003_corrected_replay",
                "source": source,
                "min": round(float(part["auroc"].min()), 4),
                "max": round(float(part["auroc"].max()), 4),
            }
        )
    for surface, part in e005_mean.groupby("surface_id"):
        ranges.append(
            {
                "cohort": "E005_primary",
                "source": surface.replace("__measured", ""),
                "min": round(float(part["auroc"].min()), 4),
                "max": round(float(part["auroc"].max()), 4),
            }
        )

    payload = {
        "schema_version": 1,
        "note": (
            "Derived from the released metric-level tables at build time. "
            "reproduce.py recomputes each value and compares it here."
        ),
        "fully_random_deltas": sorted(deltas, key=lambda row: (row["model"], row["surface"])),
        "direction_stability": sorted(stability, key=lambda row: row["model"]),
        "shortcut_deltas": sorted(shortcut_rows, key=lambda row: row["surface"]),
        "sign_agreement": sorted(
            agreement_rows, key=lambda row: (row["surface"], row["metric"])
        ),
        "vidal_intervals": sorted(interval_rows, key=lambda row: (row["model"], row["seed"])),
        "external_auroc_ranges": sorted(
            ranges, key=lambda row: (row["cohort"], row["source"])
        ),
    }
    output = ROOT / "manifests/EXPECTED_VALUES.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"  fully_random_deltas: {len(deltas)}")
    print(f"  direction_stability: {len(stability)}")
    print(f"  shortcut_deltas: {len(shortcut_rows)}")
    print(f"  sign_agreement: {len(agreement_rows)}")
    print(f"  vidal_intervals: {len(interval_rows)}")
    print(f"  external_auroc_ranges: {len(ranges)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
