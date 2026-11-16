#!/usr/bin/env python
"""Recompute the reported metric tables from the released pointwise predictions.

Every released prediction file is joined to its frozen panel, scored with the
frozen convention, and compared with the value in
``results/tables/reported_metrics.csv``.  A mismatch larger than ``--tolerance`` is
a failure: it means the released predictions and the released table do not
describe the same computation.

The join is not optional.  A prediction file carries ``row_id`` and ``score`` but
no labels; the labels live in the frozen panel.  Scoring a prediction file without
joining it to the panel would score against an all-zero label vector, so the join
is what makes the number meaningful.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppi_shift_release import panels, predictions  # noqa: E402

POINTWISE = "results/pointwise"
REPORTED = "results/tables/reported_metrics.csv"


def _iter_pointwise(root: Path):
    base = root / POINTWISE
    if not base.is_dir():
        return
    for path in sorted(base.rglob("*.csv.gz")):
        parts = path.relative_to(base).parts
        if len(parts) != 5:
            continue
        cohort, model, train_environment, seed_dir, filename = parts
        yield {
            "cohort": cohort,
            "model": model,
            "train_environment": train_environment,
            "seed": seed_dir.replace("seed-", ""),
            "test_environment": filename.split(".")[0],
            "path": path,
        }


def _panel_index(entries) -> dict[tuple[str, str], object]:
    return {(entry.cohort, entry.environment): entry for entry in entries}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    entries = panels.load_manifest(args.root)
    index = _panel_index(entries)

    reported_path = args.root / REPORTED
    reported: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    if reported_path.exists():
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

    rows = []
    problems: list[str] = []
    for record in _iter_pointwise(args.root):
        key = (
            record["cohort"],
            record["model"],
            record["train_environment"],
            record["seed"],
            record["test_environment"],
        )
        entry = index.get((record["cohort"], record["test_environment"]))
        if entry is None:
            problems.append(f"{'/'.join(key)}: no frozen panel for this cohort and environment")
            continue
        table = predictions.load_predictions(record["path"])
        summary = predictions.evaluate(args.root, entry, table, bootstrap_replicates=0)
        expected = reported.get(key)
        row = {
            **{
                name: record[name]
                for name in (
                    "cohort",
                    "model",
                    "train_environment",
                    "seed",
                    "test_environment",
                )
            },
            "n_rows": summary["n_rows"],
            "n_positive": summary["n_positive"],
            "prevalence": summary["prevalence"],
            "auroc": summary["auroc"],
            "auprc": summary["auprc"],
            "prevalence_baseline_auprc": summary["prevalence_baseline_auprc"],
            "brier": summary["brier"],
            "nll": summary["nll"],
            "ece": summary["ece"],
            "reported_auroc": "",
            "reported_auprc": "",
            "auroc_abs_diff": "",
            "auprc_abs_diff": "",
            "status": "NO_REPORTED_ROW",
        }
        if expected is not None:
            auroc_diff = abs(summary["auroc"] - float(expected["auroc"]))
            auprc_diff = abs(summary["auprc"] - float(expected["auprc"]))
            row.update(
                {
                    "reported_auroc": expected["auroc"],
                    "reported_auprc": expected["auprc"],
                    "auroc_abs_diff": auroc_diff,
                    "auprc_abs_diff": auprc_diff,
                    "status": (
                        "MATCH"
                        if max(auroc_diff, auprc_diff) <= args.tolerance
                        else "MISMATCH"
                    ),
                }
            )
            if row["status"] == "MISMATCH":
                problems.append(
                    f"{'/'.join(key)}: AUROC {summary['auroc']:.12f} vs {expected['auroc']}, "
                    f"AUPRC {summary['auprc']:.12f} vs {expected['auprc']}"
                )
        else:
            problems.append(f"{'/'.join(key)}: no reported row to compare against")
        rows.append(row)

    if not rows:
        problems.append(f"no pointwise predictions found under {args.root / POINTWISE}")

    output = args.output or (args.root / "results/tables/recomputed_metrics.csv")
    if rows:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    matches = sum(1 for row in rows if row["status"] == "MATCH")
    try:
        recorded_output = str(output.relative_to(args.root))
    except ValueError:
        recorded_output = output.name
    payload = {
        "status": "OK" if not problems else "FAILED",
        "cells": len(rows),
        "matches": matches,
        "tolerance": args.tolerance,
        "problems": problems,
        "output": recorded_output,
    }
    (args.root / "results/tables/reproduction_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"cells: {len(rows)}  matches: {matches}  tolerance: {args.tolerance}")
    if problems:
        print("\nFAILED:")
        for problem in problems[:20]:
            print(f"  - {problem}")
        if len(problems) > 20:
            print(f"  ... {len(problems) - 20} more")
    else:
        print("\nreported tables reproduce from the released predictions")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
