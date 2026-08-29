#!/usr/bin/env python3
"""Transcription-drift guard for the goto-symex verification harnesses.

A Tier-A harness (docs/roadmap/goto-symex-verification-plan.md §7.1) transcribes
the logic of a `src/goto-symex` symbol into standalone C. The transcription is
only as good as the day it was written: when the cited symbol changes, the
harness silently stops proving anything about the shipped engine.

Each harness therefore records the symbol it transcribes and a checksum of that
symbol's current definition:

    SYMEX-HARNESS-TARGET: src/goto-symex/renaming.cpp::renaming::level2t::make_assignment
    SYMEX-HARNESS-SHA256: <64 hex digits>

This script re-extracts every cited definition, re-hashes it, and fails when a
checksum no longer matches — forcing the PR author to re-transcribe, widen the
harness, or retire it (§11.5).

    drift_check.py            check every harness, exit 1 on drift
    drift_check.py --update   rewrite the recorded checksums
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

HARNESS_GLOBS = (
    "regression/esbmc/symex_*/*.c",
    "regression/esbmc/symex_*/*.cpp",
    "unit/goto-symex/*.test.cpp",
)

TARGET_RE = re.compile(r"SYMEX-HARNESS-TARGET:\s*(\S+)")
SHA_RE = re.compile(r"(SYMEX-HARNESS-SHA256:\s*)([0-9a-fA-F]{64})")
PLACEHOLDER = "0" * 64


class DriftError(Exception):
    """A cited target cannot be resolved to exactly one definition."""


def extract_definition(source: str, symbol: str) -> str:
    """Return the text of `symbol`'s definition in `source`.

    Finds the first occurrence of the symbol used as a definition — an
    occurrence whose parameter list is followed by a body rather than a
    semicolon — and returns from the start of the declaration through the
    closing brace.
    """
    pattern = re.compile(r"(?<![\w:])" + re.escape(symbol) + r"\s*\(")
    for match in pattern.finditer(source):
        body_start = _find_body(source, match.end())
        if body_start is None:
            continue
        start = _declaration_start(source, match.start())
        end = _matching_brace(source, body_start)
        return source[start:end + 1]
    raise DriftError(f"no definition of '{symbol}' found")


def _find_body(source: str, paren_start: int):
    """Index of the `{` opening the body, or None if this is a declaration."""
    depth = 1
    i = paren_start
    while i < len(source) and depth:
        depth += {"(": 1, ")": -1}.get(source[i], 0)
        i += 1
    if depth:
        return None
    while i < len(source) and source[i] not in "{;":
        i += 1
    return i if i < len(source) and source[i] == "{" else None


def _declaration_start(source: str, symbol_start: int) -> int:
    """Index just past the previous statement.

    Anchoring on the previous `;` or `}` rather than on the previous newline
    keeps the return type, the template header and any doc comment inside the
    hashed region, so a changed contract also trips the guard.
    """
    boundary = max(source.rfind(";", 0, symbol_start), source.rfind("}", 0, symbol_start))
    return len(source) - len(source[boundary + 1:].lstrip())


def _matching_brace(source: str, brace_start: int) -> int:
    depth = 0
    for i in range(brace_start, len(source)):
        depth += {"{": 1, "}": -1}.get(source[i], 0)
        if not depth:
            return i
    raise DriftError("unbalanced braces while extracting the definition")


def region_digest(root: Path, target: str) -> str:
    """sha256 of the definition cited by `<path>::<symbol>`."""
    if "::" not in target:
        raise DriftError(f"malformed target '{target}', expected <path>::<symbol>")
    rel_path, symbol = target.split("::", 1)
    path = root / rel_path
    if not path.is_file():
        raise DriftError(f"cited file '{rel_path}' does not exist")
    definition = extract_definition(path.read_text(encoding="utf-8"), symbol)
    normalised = "\n".join(line.rstrip() for line in definition.splitlines())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def check_harness(root: Path, harness: Path, update: bool) -> list:
    """Verify (or refresh) one harness. Returns a list of problem strings."""
    text = harness.read_text(encoding="utf-8")
    targets = TARGET_RE.findall(text)
    recorded = SHA_RE.findall(text)
    rel = harness.relative_to(root)

    if len(targets) != len(recorded):
        return [f"{rel}: {len(targets)} TARGET line(s) but {len(recorded)} SHA256 line(s)"]

    problems = []
    digests = []
    for target in targets:
        try:
            digests.append(region_digest(root, target))
        except DriftError as err:
            problems.append(f"{rel}: {err}")
    if problems:
        return problems

    if update:
        remaining = iter(digests)
        harness.write_text(SHA_RE.sub(lambda m: m.group(1) + next(remaining), text),
                           encoding="utf-8")
        return []

    for target, digest, (_, old) in zip(targets, digests, recorded):
        if old.lower() != digest:
            what = "unrecorded" if old == PLACEHOLDER else f"was {old[:12]}…"
            problems.append(f"{rel}: '{target}' drifted ({what}, now {digest[:12]}…)")
    return problems


def main() -> int:
    """Check (or refresh) every harness; return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update",
                        action="store_true",
                        help="rewrite recorded checksums instead of failing on drift")
    parser.add_argument("--root",
                        type=Path,
                        default=Path(__file__).resolve().parents[3],
                        help="repository root (default: inferred from this script's location)")
    args = parser.parse_args()

    harnesses = [
        p for glob in HARNESS_GLOBS for p in sorted(args.root.glob(glob))
        if TARGET_RE.search(p.read_text(encoding="utf-8"))
    ]
    problems = []
    for harness in harnesses:
        problems += check_harness(args.root, harness, args.update)
    checked = len(harnesses)

    for problem in problems:
        print(f"DRIFT: {problem}", file=sys.stderr)
    verb = "refreshed" if args.update else "checked"
    print(f"{verb} {checked} harness(es) citing src/goto-symex")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
