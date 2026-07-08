#!/usr/bin/env python3
"""Aggregate clang sanitizer log files into a deduplicated report.

Reads every ``sanitizer.<pid>`` file in ``$ESBMC_SANITIZER_LOG_DIR`` (or the
directory passed on the command line), groups findings by ``(sanitizer,
kind, first-source-frame)`` and writes a Markdown summary. When the
environment variable ``GITHUB_STEP_SUMMARY`` is set, the summary is
appended to that file as well so it shows up on the GitHub Actions job
page.

Exit status:
    0 — no findings, or all findings are silenced by their respective
        suppression files.
    1 — at least one un-suppressed finding remains.

The runtime suppressions wired by ``common-options.sh`` already filter
matched entries out of the log files; this script's exit status reflects
what survived that filter. It is therefore safe to mark the CI step as
``continue-on-error: true`` during bring-up and let the artifact upload
carry the raw logs forward for triage.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List


# UBSan emits one line per finding. The location's column suffix is
# optional on some runtimes, and the line is not always anchored to a
# line start (CTest can interleave stderr with the test runner's own
# output), so we re.search the discriminator anywhere on the line.
#   src/util/foo.cpp:42:7: runtime error: signed integer overflow: ...
_UBSAN_RE = re.compile(
    r"(?P<loc>[^:\s]+:\d+(?::\d+)?): runtime error: (?P<msg>.+)$")

# ASan/LSan reports begin with a header line containing the tool and a
# kind identifier. The kind is the first token after the colon; ASan
# sometimes appends parenthesised qualifiers (e.g.
# "alloc-dealloc-mismatch (operator new vs free) on ...") which we discard.
#   ==1234==ERROR: AddressSanitizer: heap-buffer-overflow on address ...
_HEADER_RE = re.compile(
    r"==\d+==ERROR: (?P<tool>AddressSanitizer|LeakSanitizer): (?P<kind>\S+)")

# Stack frames look like:
#   #0 0xdeadbeef in symbol_name path/to/file.cc:42:3
_FRAME_RE = re.compile(
    r"^\s*#\d+\s+0x[0-9a-fA-F]+\s+in\s+(?P<sym>\S+)"
    r"(?:\s+(?P<loc>\S+:\d+(?::\d+)?))?")


@dataclass(frozen=True)
class Finding:
    """Single sanitizer finding (already comparable; used as dedup key)."""
    tool: str
    kind: str
    location: str


_ALLOCATOR_SYM_PREFIXES = (
    "malloc",
    "calloc",
    "realloc",
    "operator",  # operator new / operator delete
    "__interceptor_",
)


def _is_allocator(sym: str) -> bool:
    """True if ``sym`` names a libc / libstdc++ allocator interceptor
    that should be skipped when picking the user-frame for dedup."""
    return any(sym.startswith(p) for p in _ALLOCATOR_SYM_PREFIXES)


def _first_user_frame(line_iter: Iterator[str]) -> str:
    """Consume frames from ``line_iter`` until the first one outside
    an allocator interceptor is found.

    Tolerates blank lines that precede the first frame (LSan inserts
    one between the header and the leak block) and stops at the first
    blank line *after* at least one frame has been seen. Returns the
    location (``file:line[:col]``) when available, else the symbol,
    else the literal ``"<no frame>"`` when the iterator runs dry.
    """
    fallback = ""
    saw_frame = False
    for line in line_iter:
        if not line.strip():
            if saw_frame:
                break
            continue
        m = _FRAME_RE.match(line)
        if not m:
            continue
        saw_frame = True
        loc = m.group("loc") or ""
        sym = m.group("sym")
        if not fallback:
            fallback = loc or sym
        if _is_allocator(sym):
            continue
        return loc or sym
    return fallback or "<no frame>"


# Skip log files larger than this — sanitizer dumps with no suppressions
# can run to hundreds of MB on hot loops; pulling such a file into memory
# would OOM the runner before the report is written.
_MAX_LOG_BYTES = 64 * 1024 * 1024


def parse_log(path: Path) -> List[Finding]:
    """Stream a single sanitizer log file, returning every finding."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        print(f"warning: cannot stat {path}: {exc}", file=sys.stderr)
        return []
    if size > _MAX_LOG_BYTES:
        print(f"warning: skipping {path} ({size} bytes > "
              f"{_MAX_LOG_BYTES} cap)",
              file=sys.stderr)
        return []
    try:
        fh = path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"warning: cannot read {path}: {exc}", file=sys.stderr)
        return []
    out: List[Finding] = []
    with fh:
        line_iter = iter(fh)
        for line in line_iter:
            ubsan = _UBSAN_RE.search(line)
            if ubsan:
                kind = ubsan.group("msg").split(":", 1)[0].strip()
                out.append(Finding("UBSan", kind, ubsan.group("loc")))
                continue
            header = _HEADER_RE.search(line)
            if header:
                tool = "ASan" if header.group(
                    "tool") == "AddressSanitizer" else "LSan"
                out.append(
                    Finding(tool, header.group("kind"),
                            _first_user_frame(line_iter)))
    return out


def collect(log_dir: Path) -> List[Finding]:
    """Return every finding in every ``sanitizer.*`` file under log_dir."""
    if not log_dir.is_dir():
        return []
    findings: List[Finding] = []
    for entry in sorted(log_dir.iterdir()):
        if entry.is_file() and entry.name.startswith("sanitizer."):
            findings.extend(parse_log(entry))
    return findings


def render_markdown(findings: Iterable[Finding]) -> str:
    """Render a Markdown table summarising the (deduplicated) findings."""
    counts: Counter = Counter(findings)
    if not counts:
        return "## Sanitizer findings\n\nNo findings.\n"
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    lines = [
        "## Sanitizer findings",
        "",
        f"{len(counts)} unique finding(s); {sum(counts.values())} total occurrence(s).",
        "",
        "| Count | Tool | Kind | Location |",
        "| ----: | :--- | :--- | :------- |",
    ]
    for finding, count in rows:
        lines.append(
            f"| {count} | {finding.tool} | {finding.kind} | `{finding.location}` |"
        )
    lines.append("")
    return "\n".join(lines)


def write_summary(report: str) -> None:
    """Write report to stdout and, when running under GitHub Actions, to
    the job-summary file pointed at by ``$GITHUB_STEP_SUMMARY``."""
    sys.stdout.write(report)
    if not report.endswith("\n"):
        sys.stdout.write("\n")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(report)
            if not report.endswith("\n"):
                fh.write("\n")


def main(argv: List[str] | None = None) -> int:
    """CLI entry point. See module docstring for exit-status semantics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_dir",
        nargs="?",
        default=os.environ.get("ESBMC_SANITIZER_LOG_DIR"),
        help="directory of sanitizer.<pid> log files "
        "(defaults to $ESBMC_SANITIZER_LOG_DIR)",
    )
    args = parser.parse_args(argv)
    if not args.log_dir:
        parser.error(
            "log_dir is required (positional or via $ESBMC_SANITIZER_LOG_DIR)")
    findings = collect(Path(args.log_dir))
    write_summary(render_markdown(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
