#!/usr/bin/env python
"""Evaluate a new model's predictions against the frozen panels.

Example
-------
    # Normal unshuffled 1024-residue Gold evaluation (default: --cohort E005_core)
    python scripts/evaluate_predictions.py \
        --predictions my_predictions.csv \
        --panel uniform_partner \
        --output results/my_model

    # Partner-shuffled 1024-residue control evaluation (explicit --cohort E005_primary)
    python scripts/evaluate_predictions.py \
        --predictions my_shuffled_predictions.csv \
        --cohort E005_primary \
        --panel uniform_partner \
        --output results/my_model_shuffled

The prediction file needs ``row_id`` and ``score`` columns.  If it carries a
``panel`` or ``environment`` column, each value is evaluated against the matching
frozen panel; otherwise ``--panel`` selects one panel explicitly.

The command refuses incomplete, duplicated, unknown or out-of-range input rather
than repairing it, because a repaired file produces a number that cannot be
compared with the reported tables.
"""

from __future__ import annotations

import argparse
import csv
import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppi_shift_release import panels, predictions  # noqa: E402

# The manifest names the cohorts after the experiment that produced them; the
# short aliases are what a reader types.  Ordinary new-model evaluation defaults
# to ``E005_core`` (unshuffled 1024-residue Gold).  ``E005_primary`` (and its
# legacy alias ``E005``) selects the broken-pair partner-shuffled control cohort.
COHORT_ALIASES = {
    "E003": "E003_corrected_replay",
    "E005": "E005_primary",
    "E005_core": "E005_core",
    "RBT": "RBT_fully_random",
}
COHORT_CHOICES = (
    "E003_corrected_replay",
    "E005_core",
    "E005_primary",
    "RBT_fully_random",
    "E003",
    "E005",
    "RBT",
)


def _split_by_panel(path: Path) -> dict[str, list[dict[str, str]]]:
    # The release ships its own pointwise predictions as .csv.gz, so this reader
    # has to accept the same container it produces.  panels.open_text handles
    # both plain CSV and gzip.
    with panels.open_text(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Prediction file has no header: {path}")
        key = next(
            (
                name
                for name in ("panel", "environment", "test_environment")
                if name in reader.fieldnames
            ),
            None,
        )
        if key is None:
            return {}
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in reader:
            grouped.setdefault(row[key], []).append(row)
    return grouped


def _write_subset(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--panel", default=None, help="panel environment name")
    parser.add_argument(
        "--cohort",
        default="E005_core",
        choices=COHORT_CHOICES,
        help="cohort to evaluate (default: E005_core unshuffled 1024-residue Gold; pass E005_primary for partner-shuffling controls)",
    )
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    args = parser.parse_args(argv)
    cohort = COHORT_ALIASES.get(args.cohort, args.cohort)

    entries = panels.load_manifest(args.root)
    available = {
        (entry.cohort, entry.split, entry.environment): entry for entry in entries
    }
    grouped = _split_by_panel(args.predictions)

    if grouped:
        targets = []
        for environment in sorted(grouped):
            key = (cohort, args.split, environment)
            if key not in available:
                raise SystemExit(
                    f"prediction file names panel {environment!r}, which is not a released "
                    f"{cohort}/{args.split} panel. Available: "
                    f"{sorted(e for c, s, e in available if (c, s) == (cohort, args.split))}"
                )
            targets.append((available[key], grouped[environment]))
    else:
        if not args.panel:
            raise SystemExit(
                "prediction file has no panel column; pass --panel with one of "
                f"{sorted(e for c, s, e in available if (c, s) == (cohort, args.split))}"
            )
        key = (cohort, args.split, args.panel)
        if key not in available:
            raise SystemExit(f"unknown panel {args.panel!r} for {cohort}/{args.split}")
        targets = [(available[key], None)]

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry, subset in targets:
        if subset is None:
            table = predictions.load_predictions(args.predictions)
        else:
            scratch = args.output / f".subset__{entry.environment}.csv"
            _write_subset(scratch, subset)
            table = predictions.load_predictions(scratch)
            scratch.unlink()
        result = predictions.evaluate(
            args.root,
            entry,
            table,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        rows.append(result)
        print(
            f"{entry.cohort}/{entry.split}/{entry.environment}: "
            f"AUROC {result['auroc']:.6f}  AUPRC {result['auprc']:.6f}  "
            f"prevalence {result['prevalence']:.6f}  n={result['n_rows']}"
        )

    predictions.write_metrics_csv(args.output / "metrics.csv", rows)
    predictions.write_report(
        args.output / "report.json",
        {
            "predictions": str(args.predictions),
            "cohort": cohort,
            "split": args.split,
            "panels": len(rows),
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "results": rows,
            "note": (
                "AUROC and AUPRC use each panel's frozen cohort convention "
                "(score_convention), with estimator-resolved ties and no jitter. "
                "AUPRC is reported next to the prevalence baseline because AUPRC "
                "is prevalence-dependent."
            ),
        },
    )
    print(f"\nwrote {args.output / 'metrics.csv'} and {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
