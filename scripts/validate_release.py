#!/usr/bin/env python
"""Validate release integrity and anonymity.

Two independent checks:

* **Integrity** - every file listed in ``manifests/INTEGRITY.json`` exists and
  hashes to the recorded SHA256, and no listed file is missing.
* **Anonymity** - no released file carries an identifying string, whether in its
  name, its bytes, a compressed member, an image or document metadata field, or
  a string literal that Python would reconstruct at run time.

This validator ships inside the archive, so it holds **no author-specific
literal**.  It applies generic identity classes only.  The private red-team scan
supplies the sensitive literals from outside the archive through ``--patterns``,
which is why the archive can be scanned for its own author without the scanner
becoming the leak it hunts.

A failure is reported with the offending file and location.  Nothing is
rewritten.
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import bz2
import gzip
import hashlib
import io
import json
import lzma
import re
import struct
import sys
import tarfile
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path

sys.dont_write_bytecode = True

INTEGRITY = "manifests/INTEGRITY.json"
ATTRIBUTION = "manifests/THIRD_PARTY_ATTRIBUTION.json"

# Directories and suffixes that must never reach a distributed tree: bytecode
# embeds the absolute path it was compiled at, and the caches embed the same.
# Caches this package can create.  They are reported, and ``--strict-artifacts``
# turns them into a failure.
BUILD_ARTIFACT_DIRS = frozenset(
    {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".ipynb_checkpoints"}
)
BUILD_ARTIFACT_SUFFIXES = frozenset({".pyc", ".pyo", ".pyd"})

# Directories that are never distributed content and are skipped silently.  A
# reviewer who follows the README creates ``.venv`` inside the checkout, and
# third-party packages carry their own repository URLs and commit identifiers;
# scanning them would report the reviewer's environment as an anonymity leak.
# ``.git`` is skipped because its compressed objects match byte patterns by
# chance, and the anonymous history is asserted separately by
# ``scripts/build_final_anonymous_git.py``.
IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        ".env",
        ".tox",
        ".nox",
        "node_modules",
        "site-packages",
        ".eggs",
        "build",
        "dist",
        "figures_regenerated",
    }
)

# Files this package writes when it runs.  They are outputs, not distributed
# inputs, and a failed run records its own findings in them.
IGNORED_FILES = frozenset(
    {
        "results/tables/reproduction_report.json",
        "results/tables/validation_report.json",
        "manifests/validation_report.json",
    }
)

# Generic identity classes.  None of these names an author, an institution or a
# machine: each describes a *shape* that identifies whoever produced the file.
GENERIC_PATTERNS: tuple[tuple[str, str], ...] = (
    ("email_address", r"[\w.+-]+@[\w-]+(?:[.][\w-]+)*[.][A-Za-z]{2,24}"),
    ("posix_home_path", r"/(?:home|Users)/[A-Za-z0-9._-]+/"),
    (
        "windows_user_path",
        r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/mnt/[a-z]/Users/)[A-Za-z0-9._-]+",
    ),
    ("wsl_unc_path", r"\\\\wsl(?:\.localhost|\$)[\\/]"),
    ("repository_url", r"(?:github|gitlab|bitbucket|codeberg)\.[A-Za-z0-9.-]+/[A-Za-z0-9._-]+/"),
    ("private_key_block", r"-----BEGIN [A-Z ]{0,20}PRIVATE KEY-----"),
    (
        "cloud_credential",
        r"(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9-]{10,})",
    ),
    (
        "credential_assignment",
        r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*"
        r"['\"][^'\"]{8,}['\"]",
    ),
    (
        "remote_shell_target",
        r"\b(?:ssh|scp|rsync|sftp)\b[^\n]{0,80}?[A-Za-z0-9._-]+@[A-Za-z0-9._-]+",
    ),
    # A SHA-1 is 40 hex characters and is what a Git commit identifier looks
    # like.  Content hashes in this release are SHA-256 (64 hex), so the two do
    # not collide.
    ("commit_identifier", r"\b[0-9a-f]{40}\b"),
)

# Suffixes whose bytes are scanned as text.  Everything else is scanned only
# through its metadata, so a compressed blob is not mistaken for prose.
TEXT_SUFFIXES = frozenset(
    {
        ".py", ".md", ".txt", ".json", ".csv", ".tsv", ".yml", ".yaml", ".cfg",
        ".toml", ".ini", ".rst", ".tex", ".bib", ".html", ".xml", ".sh", ".cff",
        ".gitignore", ".gitattributes", "",
    }
)

ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip", ".gz", ".bz2", ".xz", ".lzma")

# Compressed numeric payloads.  They carry scores and sequences, never prose,
# and the builder scans their decompressed bytes before the archive is written.
PAYLOAD_SUFFIXES = frozenset({".gz", ".npy", ".npz", ".pt", ".bin", ".zip", ".bz2", ".xz"})

MAX_ARCHIVE_DEPTH = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integrity_entries(manifest: dict) -> list[tuple[str, dict]]:
    """Accept both the list form and the mapping form of ``files``."""
    files = manifest.get("files", [])
    if isinstance(files, dict):
        return sorted(files.items())
    return sorted((record["path"], record) for record in files)


# --------------------------------------------------------------- payload access


def _decompress(name: str, data: bytes) -> bytes | None:
    try:
        if name.endswith(".gz"):
            return gzip.decompress(data)
        if name.endswith(".bz2"):
            return bz2.decompress(data)
        if name.endswith((".xz", ".lzma")):
            return lzma.decompress(data)
    except (OSError, EOFError, lzma.LZMAError):
        return None
    return None


def _strip_archive_suffix(label: str) -> str:
    """Remove one archive suffix, so a decompressed member is not re-decompressed."""
    lowered = label.lower()
    for suffix in ARCHIVE_SUFFIXES:
        if lowered.endswith(suffix):
            return label[: -len(suffix)]
    return label


def iter_payloads(label: str, data: bytes, depth: int = 0) -> Iterator[tuple[str, bytes]]:
    """Yield ``(label, bytes)`` for a file and every member it contains.

    Compressed members are decompressed and recursed into, so a leak hidden in a
    ``.tar.gz`` inside a ``.zip`` is still reached.
    """
    if depth > MAX_ARCHIVE_DEPTH:
        return
    lowered = label.lower()
    if lowered.endswith((".tar.gz", ".tgz", ".tar")):
        try:
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    inner = handle.read()
                    yield from iter_payloads(f"{label}!{member.name}", inner, depth + 1)
        except (tarfile.TarError, OSError):
            return
        return
    if lowered.endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    inner = archive.read(info)
                    yield from iter_payloads(f"{label}!{info.filename}", inner, depth + 1)
        except (zipfile.BadZipFile, OSError):
            return
        return
    if lowered.endswith(ARCHIVE_SUFFIXES):
        inner = _decompress(lowered, data)
        if inner is None:
            return
        stripped = _strip_archive_suffix(label)
        if stripped.lower().endswith(ARCHIVE_SUFFIXES):
            yield from iter_payloads(stripped, inner, depth + 1)
        else:
            yield label, inner
        return
    yield label, data


# ------------------------------------------------------------- binary metadata


def png_text_chunks(data: bytes) -> list[tuple[str, str]]:
    """Return the textual chunks of a PNG, which carry authoring metadata."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return []
    found: list[tuple[str, str]] = []
    offset = 8
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type in (b"tEXt", b"iTXt", b"zTXt"):
            found.append((chunk_type.decode("ascii"), body.decode("latin-1", "ignore")))
        if chunk_type == b"IEND":
            break
    return found


def pdf_metadata(data: bytes) -> list[tuple[str, str]]:
    """Return the document-information and XMP fields of a PDF."""
    if not data.startswith(b"%PDF"):
        return []
    found: list[tuple[str, str]] = []
    for key in (b"Author", b"Creator", b"Producer", b"Title", b"Subject", b"Keywords"):
        for match in re.finditer(rb"/" + key + rb"\s*\(([^)]{0,400})\)", data):
            found.append((key.decode("ascii"), match.group(1).decode("latin-1", "ignore")))
    for match in re.finditer(rb"<x:xmpmeta.*?</x:xmpmeta>", data, re.DOTALL):
        found.append(("xmp", match.group(0).decode("latin-1", "ignore")))
    return found


# ------------------------------------------------------- literal reconstruction


def _fold(node: ast.AST) -> str | None:
    """Return the string a node evaluates to, or ``None`` when it is dynamic."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _fold(node.left), _fold(node.right)
        if left is not None and right is not None:
            return left + right
        return None
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                return None
        return "".join(parts)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and node.args
        and isinstance(node.args[0], (ast.List, ast.Tuple))
    ):
        parts = [_fold(element) for element in node.args[0].elts]
        if all(part is not None for part in parts):
            return "".join(part for part in parts if part is not None)
    return None


def python_literals(source: str) -> list[str]:
    """Return every string a Python module would build from literals.

    Implicit concatenation is already folded by the parser; explicit ``+``,
    f-strings without placeholders and ``"".join([...])`` are folded here.  This
    is what catches an identifier split across fragments.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    literals: list[str] = []
    for node in ast.walk(tree):
        value = _fold(node)
        if value is not None:
            literals.append(value)
    return literals


def decoded_variants(text: str) -> list[str]:
    """Return printable base64 and hex decodings of a literal, if any."""
    variants: list[str] = []
    stripped = text.strip()
    if len(stripped) >= 8 and re.fullmatch(r"[A-Za-z0-9+/=_-]+", stripped):
        padded = stripped + "=" * (-len(stripped) % 4)
        for decoder in (base64.b64decode, base64.urlsafe_b64decode):
            try:
                raw = decoder(padded)
            except (binascii.Error, ValueError):
                continue
            if raw and all(32 <= byte < 127 or byte in (9, 10, 13) for byte in raw):
                variants.append(raw.decode("ascii"))
    if len(stripped) >= 8 and len(stripped) % 2 == 0 and re.fullmatch(r"[0-9a-fA-F]+", stripped):
        try:
            raw = bytes.fromhex(stripped)
        except ValueError:
            raw = b""
        if raw and all(32 <= byte < 127 or byte in (9, 10, 13) for byte in raw):
            variants.append(raw.decode("ascii"))
    return variants


# ------------------------------------------------------------------- scanning


def load_patterns(path: Path | None) -> list[tuple[str, re.Pattern[str]]]:
    """Compile the generic classes plus any externally supplied ones."""
    compiled = [(label, re.compile(pattern)) for label, pattern in GENERIC_PATTERNS]
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected a JSON object of label -> regex")
        for label, pattern in sorted(payload.items()):
            compiled.append((f"external:{label}", re.compile(pattern)))
    return compiled


PATTERN_HINTS: dict[str, tuple[str, ...]] = {
    "email_address": ("@",),
    "posix_home_path": ("/home/", "/Users/"),
    "windows_user_path": ("Users", "users"),
    "wsl_unc_path": ("wsl",),
    "repository_url": (".com/", "github.", "gitlab.", "bitbucket.", "codeberg."),
    "private_key_block": ("PRIVATE KEY",),
    "cloud_credential": ("AKIA", "gh", "xox"),
    "credential_assignment": ("'", '"'),
    "remote_shell_target": ("@",),
}


def _scan_text(
    label: str,
    text: str,
    patterns: Sequence[tuple[str, re.Pattern[str]]],
    allow: frozenset[str],
) -> list[dict[str, str]]:
    active_patterns = []
    for name, pattern in patterns:
        hints = PATTERN_HINTS.get(name)
        if hints is not None and not any(h in text for h in hints):
            continue
        for match in pattern.finditer(text):
            if match.group(0) not in allow:
                active_patterns.append((name, pattern))
                break
    if not active_patterns:
        return []
    findings: list[dict[str, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for name, pattern in active_patterns:
            match = pattern.search(line)
            if match is None or match.group(0) in allow:
                continue
            findings.append(
                {
                    "file": label,
                    "line": str(number),
                    "kind": name,
                    "text": line.strip()[:160],
                }
            )
            break
    return findings


def _effective_suffix(label: str) -> str:
    """Suffix of the innermost member, so a decompressed CSV is scanned as text."""
    name = label.rsplit("!", 1)[-1].lower()
    for suffix in (".gz", ".bz2", ".xz", ".lzma"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return Path(name).suffix


SCORE_PAYLOAD_PREFIXES = (
    "results/pointwise/",
    "results/fully_random_pointwise/",
    "results/fully_random_external_pointwise/",
    "results/shuffle_pointwise/",
    "results/external_pointwise/",
    "results/common_512_pointwise/",
    "results/validation_pointwise/",
)


def scan_tree(
    root: Path,
    patterns: Sequence[tuple[str, re.Pattern[str]]],
    allow: frozenset[str] = frozenset(),
    *,
    skip_payloads: bool = False,
) -> tuple[list[dict[str, str]], list[str]]:
    """Scan every file under ``root``; return ``(findings, build_artifacts)``.

    ``skip_payloads`` omits compressed numeric payloads.  The release builder
    scans their decompressed members before packaging; routine integrity checks
    then verify every distributed byte against the release manifest.
    """
    findings: list[dict[str, str]] = []
    artifacts: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        parts = relative.parts
        if any(part in IGNORED_DIRS for part in parts):
            continue
        if relative.as_posix() in IGNORED_FILES:
            continue
        if any(part.endswith(".egg-info") for part in parts):
            continue
        if any(part in BUILD_ARTIFACT_DIRS for part in parts) or (
            path.suffix.lower() in BUILD_ARTIFACT_SUFFIXES
        ):
            artifacts.append(relative.as_posix())
            continue
        if not path.is_file():
            continue
        if skip_payloads and path.suffix.lower() in PAYLOAD_SUFFIXES:
            continue
        label = relative.as_posix()

        for name, pattern in patterns:
            match = pattern.search(label)
            if match is not None and match.group(0) not in allow:
                findings.append(
                    {"file": label, "line": "0", "kind": name, "text": "filename"}
                )
                break

        if label.startswith(SCORE_PAYLOAD_PREFIXES) and path.suffix.lower() == ".gz":
            try:
                with gzip.open(path, "rb") as gz_handle:
                    sample = gz_handle.read(65536).decode("utf-8", errors="ignore")
                findings.extend(_scan_text(label, sample, patterns, allow))
            except OSError:
                pass
            continue

        data = path.read_bytes()

        if path.suffix.lower() == ".png":
            for chunk, body in png_text_chunks(data):
                findings.extend(_scan_text(f"{label}#{chunk}", body, patterns, allow))
        elif path.suffix.lower() == ".pdf":
            for field, body in pdf_metadata(data):
                findings.extend(_scan_text(f"{label}#{field}", body, patterns, allow))

        for member, payload in iter_payloads(label, data):
            if _effective_suffix(member) in TEXT_SUFFIXES:
                text = payload.decode("utf-8", errors="ignore")
                findings.extend(_scan_text(member, text, patterns, allow))
            if member.endswith(".py"):
                source = payload.decode("utf-8", errors="ignore")
                for literal in python_literals(source):
                    for variant in (literal, *decoded_variants(literal)):
                        findings.extend(
                            _scan_text(f"{member}#literal", variant, patterns, allow)
                        )
    return findings, artifacts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-anonymity", action="store_true")
    parser.add_argument(
        "--skip-integrity",
        action="store_true",
        help="scan for identifying content only; do not re-hash every file",
    )
    parser.add_argument(
        "--skip-payloads",
        action="store_true",
        help=(
            "skip compressed numeric payloads (.gz/.npy/.npz/.pt/.bin/.zip); "
            "the release builder scans those byte for byte"
        ),
    )
    parser.add_argument(
        "--patterns",
        type=Path,
        default=None,
        help="JSON object of extra label -> regex patterns; never shipped",
    )
    parser.add_argument(
        "--allow",
        type=Path,
        default=None,
        help="JSON list of exact strings that are known-safe and must not be flagged",
    )
    parser.add_argument(
        "--strict-artifacts",
        action="store_true",
        help="treat build caches and bytecode as a failure rather than a warning",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    problems: list[str] = []
    integrity_path = args.root / INTEGRITY
    checked = 0
    if args.skip_integrity:
        manifest: dict = {"files": {}}
    elif not integrity_path.exists():
        problems.append(f"missing {INTEGRITY}")
        manifest = {"files": {}}
    else:
        manifest = json.loads(integrity_path.read_text(encoding="utf-8"))
        for relative, record in _integrity_entries(manifest):
            path = args.root / relative
            if not path.exists():
                problems.append(f"missing file: {relative}")
                continue
            if sha256_file(path) != record["sha256"]:
                problems.append(f"hash mismatch: {relative}")
                continue
            checked += 1

    anonymity_hits: list[dict[str, str]] = []
    artifacts: list[str] = []
    if not args.skip_anonymity:
        # Third-party attribution is required, not forbidden.  The panel rows
        # record the upstream repository and revision of the method each
        # negative sampler reproduces, so the generic repository_url and
        # commit_identifier patterns match them once per row.  The shipped
        # allow-list names those exact strings; --allow overrides it.
        allow_path = args.allow or (args.root / ATTRIBUTION)
        allow = frozenset()
        if allow_path.exists():
            allow = frozenset(json.loads(allow_path.read_text(encoding="utf-8")))
        anonymity_hits, artifacts = scan_tree(
            args.root,
            load_patterns(args.patterns),
            allow,
            skip_payloads=args.skip_payloads,
        )
        if anonymity_hits:
            problems.append(f"{len(anonymity_hits)} anonymity findings")
        if artifacts and args.strict_artifacts:
            problems.append(f"{len(artifacts)} build artifacts present")

    payload = {
        "status": "OK" if not problems else "FAILED",
        "files_checked": checked,
        "files_listed": len(_integrity_entries(manifest)),
        "anonymity_findings": anonymity_hits,
        "build_artifacts": artifacts,
        "problems": problems,
    }
    output = args.output or (args.root / "manifests/validation_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"integrity: {checked}/{payload['files_listed']} files verified")
    print(f"anonymity findings: {len(anonymity_hits)}")
    print(f"build artifacts: {len(artifacts)}")
    if problems:
        print("\nFAILED:")
        for problem in problems[:20]:
            print(f"  - {problem}")
        for hit in anonymity_hits[:10]:
            print(f"  {hit['file']}:{hit['line']} [{hit['kind']}] {hit['text']}")
    else:
        print("\nrelease OK")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
