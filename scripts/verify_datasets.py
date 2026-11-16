#!/usr/bin/env python
"""Verify that every released panel reconstructs from the released sequences.

Checks, for all panels in the manifest:

* the panel file hash matches the manifest;
* the row count and positive count match the manifest;
* row identifiers are unique;
* labels are binary;
* every row resolves to two sequences that hash to the stored digests;
* the six negative environments are distinguishable by their own column;
* synthetic negatives are labelled as such and never as assay-validated negatives.

Any failure is reported and the exit status is non-zero.  Nothing is repaired.
"""

from __future__ import annotations

import argparse
import json
import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppi_shift_release import panels  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--skip-sequences",
        action="store_true",
        help="skip the sequence join (fast structural check only)",
    )
    args = parser.parse_args(argv)

    problems: list[str] = []
    entries = panels.load_manifest(args.root)
    if not entries:
        problems.append("panel manifest is empty")

    registry: dict[str, object] = {}
    sequences: dict[str, str] = {}
    if not args.skip_sequences:
        registry = panels.verify_sequence_registry(args.root)
        if registry["mismatched"]:
            problems.append(
                f"{registry['mismatched']} registry sequences do not hash to their key"
            )
        sequences = panels.load_sequences(args.root)

    panel_reports = []
    for entry in entries:
        report = panels.verify_panel(args.root, entry)
        if report["status"] != "OK":
            problems.append(f"{entry.path}: {report['status']}")
        if report.get("rows") != report.get("expected_rows"):
            problems.append(
                f"{entry.path}: {report.get('rows')} rows, manifest says "
                f"{report.get('expected_rows')}"
            )
        if report.get("n_positive") != report.get("expected_n_positive"):
            problems.append(f"{entry.path}: positive count differs from manifest")
        if report.get("duplicate_row_ids"):
            problems.append(f"{entry.path}: duplicate row identifiers")
        if not report.get("labels_binary", False):
            problems.append(f"{entry.path}: labels are not binary")
        if sequences:
            try:
                resolved = panels.resolve_panel(args.root, entry, sequences)
            except (KeyError, ValueError) as error:
                problems.append(str(error))
            else:
                if len(resolved) != entry.rows:
                    problems.append(f"{entry.path}: resolved {len(resolved)} rows")
        panel_reports.append(report)

    environment_report = panels.check_environments_distinguishable(args.root, entries)
    if not environment_report["all_present"]:
        problems.append(
            "negative environments are not all distinguishable: "
            f"found {environment_report['environments']}"
        )

    semantics = sorted({entry.evidence_semantics for entry in entries})
    if "synthetic_unobserved_pair" in semantics:
        pass
    else:
        problems.append(
            "no panel declares synthetic_unobserved_pair semantics; the release must not "
            "present synthetic negatives as assay-validated negatives"
        )

    payload = {
        "status": "OK" if not problems else "FAILED",
        "panels": len(entries),
        "sequence_registry": registry,
        "negative_environments": environment_report,
        "evidence_semantics": semantics,
        "problems": problems,
        "panel_reports": panel_reports,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(f"panels: {len(entries)}")
    if registry:
        print(f"sequences: {registry['sequences']} ({registry['total_residues']} residues)")
    print(f"negative environments: {', '.join(environment_report['environments'])}")
    print(f"evidence semantics: {', '.join(semantics)}")
    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\ndatasets OK")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
