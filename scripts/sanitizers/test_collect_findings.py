#!/usr/bin/env python3
"""Unit tests for collect_findings.py.

Exercises every branch with real fixture logs (no mocks).

Run from this directory:
    python3 -m unittest test_collect_findings.py
"""
# Standard unittest patterns trip several pylint checks: setUp/tearDown owns
# the temp dir lifecycle (consider-using-with), tests legitimately exercise
# the module's private frame helper (protected-access), and the sys.path
# bootstrap precedes the local import (wrong-import-position).
# pylint: disable=consider-using-with,protected-access,wrong-import-position
# pylint: disable=missing-function-docstring,missing-class-docstring

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_findings as cf  # noqa: E402  # pyright: ignore[reportMissingImports]


def _write(dir_path: Path, name: str, content: str) -> Path:
    """Helper: write a sanitizer log fixture and return its path."""
    p = dir_path / name
    p.write_text(content)
    return p


# Realistic UBSan one-liner (the column part is optional in the wild).
UBSAN_LINE = (
    "src/util/foo.cpp:42:7: runtime error: signed integer overflow: "
    "2147483647 + 1 cannot be represented in type 'int'\n")

# Realistic multi-line ASan report.
ASAN_REPORT = """\
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0xdeadbeef at pc 0xabc
READ of size 4 at 0xdeadbeef thread T0
    #0 0x123456 in foo() src/foo.cc:42:3
    #1 0x789abc in main src/main.cc:10:5
"""

# Realistic LSan report. First #0 is an interceptor (malloc), must be skipped.
LSAN_REPORT = """\
==99999==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 16 byte(s) in 1 object(s) allocated from:
    #0 0xaaaa in malloc /usr/lib/llvm/asan_interceptors.cpp:707:3
    #1 0xbbbb in build_payload() src/payload.cc:88:13
    #2 0xcccc in main src/main.cc:3:5
"""


def _tail_iter(text: str):
    """Yield every line of ``text`` after the first (which is the
    sanitizer header) — matching what parse_log feeds to
    _first_user_frame in production."""
    lines = text.splitlines()
    return iter(lines[1:])


class FrameParsing(unittest.TestCase):
    """_first_user_frame walks past allocator interceptors."""

    def test_skips_allocator_interceptor(self):
        loc = cf._first_user_frame(_tail_iter(LSAN_REPORT))
        self.assertEqual(loc, "src/payload.cc:88:13")

    def test_returns_first_frame_when_no_interceptor(self):
        loc = cf._first_user_frame(_tail_iter(ASAN_REPORT))
        self.assertEqual(loc, "src/foo.cc:42:3")

    def test_returns_symbol_when_location_missing(self):
        # Synthesised: a frame with no file:line suffix.
        loc = cf._first_user_frame(iter(["    #0 0xa in mystery_symbol", ""]))
        self.assertEqual(loc, "mystery_symbol")

    def test_falls_back_when_no_frame_at_all(self):
        loc = cf._first_user_frame(iter(["", ""]))
        self.assertEqual(loc, "<no frame>")

    def test_allocator_prefix_matching(self):
        # Reviewer's case: `operator new(unsigned long)` symbolises with
        # `sym == "operator"` because of the \S+ capture; the prefix
        # check must still treat it as an allocator.
        self.assertTrue(cf._is_allocator("operator"))
        self.assertTrue(cf._is_allocator("__interceptor_malloc"))
        self.assertTrue(cf._is_allocator("malloc"))
        self.assertFalse(cf._is_allocator("my_function"))


class ParseLog(unittest.TestCase):
    """parse_log extracts UBSan, ASan, and LSan findings from a file."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_ubsan_oneliner(self):
        f = _write(self.dir, "sanitizer.100", UBSAN_LINE)
        findings = cf.parse_log(f)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].tool, "UBSan")
        self.assertEqual(findings[0].kind, "signed integer overflow")
        self.assertEqual(findings[0].location, "src/util/foo.cpp:42:7")

    def test_asan_report(self):
        f = _write(self.dir, "sanitizer.101", ASAN_REPORT)
        findings = cf.parse_log(f)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].tool, "ASan")
        self.assertEqual(findings[0].kind, "heap-buffer-overflow")
        self.assertEqual(findings[0].location, "src/foo.cc:42:3")

    def test_lsan_report(self):
        f = _write(self.dir, "sanitizer.102", LSAN_REPORT)
        findings = cf.parse_log(f)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].tool, "LSan")
        # LSan headers have no quantitative tail; the regex grabs everything
        # after the tool name.
        self.assertIn("detected", findings[0].kind.lower() + " memory leaks")
        self.assertEqual(findings[0].location, "src/payload.cc:88:13")

    def test_mixed_file(self):
        f = _write(self.dir, "sanitizer.103", UBSAN_LINE + ASAN_REPORT)
        findings = cf.parse_log(f)
        tools = sorted(f.tool for f in findings)
        self.assertEqual(tools, ["ASan", "UBSan"])

    def test_unreadable_file_returns_empty(self):
        # Point parse_log at a directory rather than a file; open raises.
        findings = cf.parse_log(self.dir)
        self.assertEqual(findings, [])

    def test_asan_header_with_parenthesised_qualifier(self):
        # Regression: alloc-dealloc-mismatch headers carry a parenthesised
        # qualifier between the kind and the address. The previous regex
        # required ` on ` or ` detected` directly after the kind, dropping
        # this entire class of findings — the silent-miss the design was
        # meant to prevent.
        report = ("==1==ERROR: AddressSanitizer: alloc-dealloc-mismatch "
                  "(operator new vs free) on 0xdeadbeef\n"
                  "    #0 0x1 in operator new(unsigned long) /lib/new.cc:9\n"
                  "    #1 0x2 in user_fn() src/code.cc:55:1\n")
        f = _write(self.dir, "sanitizer.200", report)
        findings = cf.parse_log(f)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].kind, "alloc-dealloc-mismatch")
        self.assertEqual(findings[0].location, "src/code.cc:55:1")

    def test_ubsan_line_with_leading_prefix(self):
        # CTest can prefix lines with the test name; the parser must
        # still pick the finding up.
        line = "[123/456] " + UBSAN_LINE
        f = _write(self.dir, "sanitizer.201", line)
        findings = cf.parse_log(f)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].tool, "UBSan")

    def test_oversized_log_is_skipped(self):
        big = self.dir / "sanitizer.300"
        # Cap is 64 MiB; produce a file just over it cheaply via seek.
        with big.open("wb") as fh:
            fh.seek(cf._MAX_LOG_BYTES + 1)
            fh.write(b"\0")
        findings = cf.parse_log(big)
        self.assertEqual(findings, [])

    def test_duplicates_within_one_log(self):
        # Two identical UBSan lines in one log must yield two Findings
        # (Counter then collapses them); pin the parser contract.
        f = _write(self.dir, "sanitizer.301", UBSAN_LINE + UBSAN_LINE)
        findings = cf.parse_log(f)
        self.assertEqual(len(findings), 2)
        self.assertEqual(findings[0], findings[1])


class Collect(unittest.TestCase):
    """collect walks the directory in sorted order, ignores other files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_directory(self):
        self.assertEqual(cf.collect(Path("/no/such/path/xyz")), [])

    def test_skips_non_sanitizer_files(self):
        _write(self.dir, "unrelated.log", UBSAN_LINE)
        self.assertEqual(cf.collect(self.dir), [])

    def test_picks_up_sanitizer_prefix_only(self):
        _write(self.dir, "sanitizer.100", UBSAN_LINE)
        _write(self.dir, "sanitizer.200", ASAN_REPORT)
        findings = cf.collect(self.dir)
        self.assertEqual(len(findings), 2)


class Render(unittest.TestCase):
    """render_markdown dedups and sorts by descending count."""

    def test_empty(self):
        out = cf.render_markdown([])
        self.assertIn("No findings", out)

    def test_dedup_counts(self):
        f = cf.Finding("UBSan", "shift", "src/a.cc:1")
        out = cf.render_markdown([f, f, f])
        self.assertIn("| 3 | UBSan | shift | `src/a.cc:1` |", out)
        self.assertIn("1 unique finding(s); 3 total occurrence(s).", out)

    def test_sort_descending_then_lex(self):
        many = cf.Finding("UBSan", "shift", "src/a.cc:1")
        once = cf.Finding("ASan", "heap-buffer-overflow", "src/b.cc:1")
        out = cf.render_markdown([many, many, once])
        # The "many" row appears before the "once" row in the table body.
        many_pos = out.index("| 2 | UBSan |")
        once_pos = out.index("| 1 | ASan |")
        self.assertLess(many_pos, once_pos)


class WriteSummary(unittest.TestCase):
    """write_summary appends to $GITHUB_STEP_SUMMARY when set."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self._prior = os.environ.get("GITHUB_STEP_SUMMARY")

    def tearDown(self):
        if self._prior is None:
            os.environ.pop("GITHUB_STEP_SUMMARY", None)
        else:
            os.environ["GITHUB_STEP_SUMMARY"] = self._prior
        self.tmp.cleanup()

    def test_appends_when_env_set(self):
        summary_file = self.dir / "summary.md"
        os.environ["GITHUB_STEP_SUMMARY"] = str(summary_file)
        cf.write_summary("hello world")
        self.assertEqual(summary_file.read_text(), "hello world\n")
        # Second call appends rather than overwrites.
        cf.write_summary("again\n")
        self.assertEqual(summary_file.read_text(), "hello world\nagain\n")

    def test_no_env_means_stdout_only(self):
        os.environ.pop("GITHUB_STEP_SUMMARY", None)
        # Should not raise.
        cf.write_summary("stdout-only\n")


class Main(unittest.TestCase):
    """CLI: positional arg, env-var fallback, exit codes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self._prior_env = os.environ.get("ESBMC_SANITIZER_LOG_DIR")
        self._prior_sum = os.environ.get("GITHUB_STEP_SUMMARY")
        os.environ.pop("ESBMC_SANITIZER_LOG_DIR", None)
        os.environ.pop("GITHUB_STEP_SUMMARY", None)

    def tearDown(self):
        for k, v in (
            ("ESBMC_SANITIZER_LOG_DIR", self._prior_env),
            ("GITHUB_STEP_SUMMARY", self._prior_sum),
        ):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def test_exit_zero_when_clean(self):
        # Empty directory → exit 0.
        rc = cf.main([str(self.dir)])
        self.assertEqual(rc, 0)

    def test_exit_one_when_findings(self):
        _write(self.dir, "sanitizer.500", UBSAN_LINE)
        rc = cf.main([str(self.dir)])
        self.assertEqual(rc, 1)

    def test_env_fallback(self):
        _write(self.dir, "sanitizer.501", UBSAN_LINE)
        os.environ["ESBMC_SANITIZER_LOG_DIR"] = str(self.dir)
        rc = cf.main([])
        self.assertEqual(rc, 1)

    def test_missing_arg_errors(self):
        with self.assertRaises(SystemExit) as ctx:
            cf.main([])
        # argparse calls parser.error → SystemExit(2)
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
