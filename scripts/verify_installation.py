#!/usr/bin/env python
"""Check that the release can be imported and that its dependencies are present.

Exits non-zero with a precise message rather than degrading to a partial state.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys

sys.dont_write_bytecode = True
from pathlib import Path

REQUIRED = {
    "numpy": "1.26",
    "pandas": "2.0",
    "sklearn": "1.3",
    "scipy": "1.10",
    "matplotlib": "3.7",
}
MINIMUM_PYTHON = (3, 10)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args(argv)

    problems: list[str] = []
    if sys.version_info < MINIMUM_PYTHON:
        problems.append(
            f"Python {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]}+ required, found "
            f"{platform.python_version()}"
        )

    versions: dict[str, str] = {}
    for module_name in REQUIRED:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            problems.append(f"missing dependency: {module_name}")
            continue
        versions[module_name] = getattr(module, "__version__", "unknown")

    source = args.root / "src"
    if not source.is_dir():
        problems.append(f"missing source directory: {source}")
    else:
        sys.path.insert(0, str(source))
        try:
            package = importlib.import_module("ppi_shift_release")
        except ImportError as error:
            problems.append(f"cannot import ppi_shift_release: {error}")
        else:
            versions["ppi_shift_release"] = package.__version__

    for relative in (
        "data/manifests/panel_manifest.json",
        "data/sequence_registry/sequences.fasta.gz",
        "manifests/INTEGRITY.json",
    ):
        if not (args.root / relative).exists():
            problems.append(f"missing release asset: {relative}")

    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "problems": problems,
        "status": "OK" if not problems else "FAILED",
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"python {payload['python']} on {payload['platform']}")
        for name, version in sorted(versions.items()):
            print(f"  {name} {version}")
        if problems:
            print("\nFAILED:")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print("\ninstallation OK")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
