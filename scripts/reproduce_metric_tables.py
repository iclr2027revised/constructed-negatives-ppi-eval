#!/usr/bin/env python
"""Recompute the metric-level claims and compare them with the manifest.

This is the metric-level counterpart of ``reproduce_results.py``. It covers the
surfaces whose prediction files are not redistributable: the two-endpoint Fully
Random arm, the Vidal/Y2H and IntAct measured-source evaluations, the symmetric
endpoint controls, and the per-seed intervals.

It does not resample anything. The intervals are read from the released per-seed
table and checked against the manifest; regenerating them would require the
pointwise predictions, which are not distributed for these surfaces.
"""

from __future__ import annotations

import argparse
import json
import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppi_shift_release import tables  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    result = tables.verify(args.root, tolerance=args.tolerance)
    output = args.output or (args.root / "results/tables/metric_level_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"claims: {result['checks']}  matched: {result['matched']}  tolerance: {args.tolerance}")
    failures = [check for check in result["detail"] if check["status"] != "MATCH"]
    if failures:
        print("\nFAILED:")
        for check in failures[:20]:
            print(
                f"  - {check['claim']}: expected {check['expected']}, "
                f"observed {check['observed']}"
            )
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} more")
        return 1
    print("\nmetric-level claims reproduce from the released tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
