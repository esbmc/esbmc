#!/usr/bin/env python3
"""Tests for the fast-lane CI scripts (esbmc/esbmc#6735).

Run with: python3 scripts/ci/ci_scripts_test.py

Everything here works on real files in a temporary directory; no mocks, per the
repository's testing rules.
"""

# Test names carry the intent, so per-method docstrings would only restate them;
# the sibling imports must follow the sys.path setup below.
# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=wrong-import-position,invalid-name

import base64
import contextlib
import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zlib
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coverage_core_set as ccs  # noqa: E402
import ctest_timings as timings  # noqa: E402
import run_selected_tests as runner  # noqa: E402
import nightly_bisect as bisect  # noqa: E402
import nightly_report as nightly  # noqa: E402
import select_tests as sel  # noqa: E402

JUNIT = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="Linux-c++" tests="3">
  <testcase name="regression/esbmc/alpha" time="1.500" status="run">
    <system-out>lots of output</system-out>
  </testcase>
  <testcase name="regression/python/beta" time="0.250" status="fail">
    <failure message="boom">trace</failure>
  </testcase>
  <testcase name="regression/python/gamma" time="0.001" status="run">
    <skipped/>
  </testcase>
</testsuite>
"""


class TempDirCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def write(self, name, text):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path


class CtestTimings(TempDirCase):

    def test_parses_time_and_status(self):
        parsed = dict((n, (s, st)) for n, s, st in timings.parse_junit(self.write("j.xml", JUNIT)))
        self.assertEqual(parsed["regression/esbmc/alpha"], (1.5, "run"))
        self.assertEqual(parsed["regression/python/beta"], (0.25, "fail"))
        self.assertEqual(parsed["regression/python/gamma"][1], "skip")

    def test_merge_keeps_untouched_entries(self):
        previous = {"tests": {"regression/z3/old": {"seconds": 9.0, "status": "run"}}}
        _, table = timings.build_table(self.write("j.xml", JUNIT), previous)
        self.assertEqual(table["tests"]["regression/z3/old"]["seconds"], 9.0)
        self.assertEqual(table["tests"]["regression/esbmc/alpha"]["seconds"], 1.5)

    def test_skip_does_not_erase_a_real_measurement(self):
        # A test skipped on this host must keep the duration another host
        # measured, or the sampler would treat it as free.
        previous = {"tests": {"regression/python/gamma": {"seconds": 42.0, "status": "run"}}}
        _, table = timings.build_table(self.write("j.xml", JUNIT), previous)
        self.assertEqual(table["tests"]["regression/python/gamma"]["seconds"], 42.0)

    def test_first_time_skip_is_not_recorded(self):
        # An unmeasured test skipped on this host must stay unmeasured, or
        # select_tests.py would floor its near-zero duration to MIN_COST
        # instead of imputing the suite median.
        _, table = timings.build_table(self.write("j.xml", JUNIT))
        self.assertNotIn("regression/python/gamma", table["tests"])

    def test_empty_report_is_an_error(self):
        empty = self.write("e.xml", '<?xml version="1.0"?><testsuite tests="0"/>')
        out = os.path.join(self.tmp, "t.json")
        self.assertEqual(timings.main(["--junit", empty, "--output", out]), 1)
        self.assertFalse(os.path.exists(out))

    def test_writes_sorted_table(self):
        out = os.path.join(self.tmp, "t.json")
        self.assertEqual(timings.main(["--junit", self.write("j.xml", JUNIT), "--output", out]), 0)
        with open(out, encoding="utf-8") as fh:
            table = json.load(fh)
        self.assertEqual(list(table["tests"]), sorted(table["tests"]))
        self.assertEqual(table["schema"], timings.SCHEMA)


class Strata(unittest.TestCase):

    def test_two_and_three_level_regression_names(self):
        self.assertEqual(sel.stratum_of("regression/esbmc/foo"), "esbmc")
        self.assertEqual(sel.stratum_of("regression/esbmc-cpp/cpp/foo"), "esbmc-cpp/cpp")
        self.assertEqual(sel.stratum_of("regression/z3/foo[bitwuzla]"), "z3")

    def test_unit_tests_are_their_own_stratum(self):
        self.assertEqual(sel.stratum_of("Interval templates division"), "unit")
        self.assertEqual(sel.stratum_of("regression/orphan"), "unit")

    def test_iso_week_format(self):
        self.assertEqual(sel.iso_week(date(2026, 8, 31)), "2026-W36")
        self.assertEqual(sel.iso_week(date(2026, 1, 1)), "2026-W01")


class Costs(unittest.TestCase):

    def test_unknown_test_takes_its_own_suites_median(self):
        names = ["regression/a/x", "regression/a/y", "regression/a/new", "regression/b/z"]
        measured = {"regression/a/x": 2.0, "regression/a/y": 4.0, "regression/b/z": 100.0}
        costs = sel.impute_costs(names, measured)
        self.assertEqual(costs["regression/a/new"], 3.0)

    def test_unknown_suite_falls_back_to_the_global_median(self):
        costs = sel.impute_costs(["regression/a/x", "regression/new/y"], {"regression/a/x": 7.0})
        self.assertEqual(costs["regression/new/y"], 7.0)

    def test_a_zero_measurement_is_floored(self):
        costs = sel.impute_costs(["regression/a/x"], {"regression/a/x": 0.0})
        self.assertEqual(costs["regression/a/x"], sel.MIN_COST)

    def test_no_measurements_at_all_still_gives_a_nonzero_cost(self):
        costs = sel.impute_costs(["regression/a/x"], {})
        self.assertEqual(costs["regression/a/x"], sel.FALLBACK_COST)


def universe(suites=6, per_suite=60):
    return [f"regression/suite{s}/test{i}" for s in range(suites) for i in range(per_suite)]


def costs_for(names):
    # Spread costs so the runtime terciles are meaningfully different.
    return {n: 0.5 + (i % 10) for i, n in enumerate(names)}


class Selection(unittest.TestCase):

    def setUp(self):
        self.names = universe()
        self.costs = costs_for(self.names)

    def test_same_week_is_reproducible(self):
        a, _ = sel.select(self.names, self.costs, [], 400.0, "2026-W36")
        b, _ = sel.select(self.names, self.costs, [], 400.0, "2026-W36")
        self.assertEqual(a, b)

    def test_a_different_week_rolls_a_different_sample(self):
        a, _ = sel.select(self.names, self.costs, [], 400.0, "2026-W36")
        b, _ = sel.select(self.names, self.costs, [], 400.0, "2026-W37")
        self.assertNotEqual(a, b)

    def test_stays_within_budget(self):
        for budget in (200.0, 400.0, 900.0):
            _, stats = sel.select(self.names, self.costs, [], budget, "2026-W36")
            self.assertLessEqual(stats["cpu_seconds"], budget, f"budget {budget}")

    def test_no_suite_goes_untested(self):
        # The whole point of stratifying. The budget here is smaller than any
        # single test, so proportional allocation alone would select nothing:
        # only the guaranteed one-per-stratum draw keeps every area covered.
        selected, _ = sel.select(self.names, self.costs, [], 1.0, "2026-W36")
        suites = {sel.stratum_of(n) for n in self.names}
        self.assertEqual({sel.stratum_of(n) for n in selected}, suites)
        self.assertEqual(len(selected), len(suites))

    def test_slow_tests_are_not_squeezed_out(self):
        selected, _ = sel.select(self.names, self.costs, [], 600.0, "2026-W36")
        slowest = max(self.costs.values())
        self.assertTrue(any(self.costs[n] >= slowest - 1 for n in selected))

    def test_core_set_is_always_present(self):
        core = ["regression/suite0/test0", "regression/suite3/test7"]
        selected, stats = sel.select(self.names, self.costs, core, 5.0, "2026-W36")
        self.assertTrue(set(core) <= set(selected))
        self.assertEqual(stats["always_run"], 2)

    def test_core_set_entries_absent_from_the_build_are_dropped(self):
        selected, stats = sel.select(self.names, self.costs, ["regression/gone/test0"], 50.0,
                                     "2026-W36")
        self.assertNotIn("regression/gone/test0", selected)
        self.assertEqual(stats["always_run"], 0)

    def test_a_bigger_budget_selects_more(self):
        small, _ = sel.select(self.names, self.costs, [], 100.0, "2026-W36")
        large, _ = sel.select(self.names, self.costs, [], 800.0, "2026-W36")
        self.assertGreater(len(large), len(small))

    def test_growing_one_suite_does_not_reshuffle_another(self):
        grown = self.names + [f"regression/suite9/test{i}" for i in range(40)]
        costs = costs_for(grown)
        base_seed = [n for n in sel.select(self.names, self.costs, [], 0.0, "2026-W36")[0]
                     if sel.stratum_of(n) == "suite0"]
        grown_seed = [n for n in sel.select(grown, costs, [], 0.0, "2026-W36")[0]
                      if sel.stratum_of(n) == "suite0"]
        self.assertEqual(base_seed, grown_seed)

    def test_a_suite_measured_entirely_at_zero_does_not_divide_by_zero(self):
        # Every test in suite0 skipped on the measuring host, so the whole
        # stratum weighs nothing and its budget share is split by zero.
        measured = {
            n: (0.0 if sel.stratum_of(n) == "suite0" else self.costs[n])
            for n in self.names
        }
        costs = sel.impute_costs(self.names, measured)
        selected, _ = sel.select(self.names, costs, [], 400.0, "2026-W36")
        self.assertTrue(any(sel.stratum_of(n) == "suite0" for n in selected))

    def test_a_duplicated_core_set_entry_is_counted_once(self):
        core = ["regression/suite0/test0", "regression/suite0/test0"]
        _, stats = sel.select(self.names, self.costs, core, 50.0, "2026-W36")
        self.assertEqual(stats["always_run"], 1)

    def test_stats_cover_every_suite(self):
        _, stats = sel.select(self.names, self.costs, [], 400.0, "2026-W36")
        self.assertEqual(sum(v["total"] for v in stats["strata"].values()), len(self.names))
        self.assertEqual(sum(v["selected"] for v in stats["strata"].values()), stats["selected"])


class SelectorCli(TempDirCase):

    def test_end_to_end(self):
        names = universe(3, 20)
        table = {
            "schema": 1,
            "tests": {n: {"seconds": c, "status": "run"} for n, c in costs_for(names).items()},
        }
        timings_path = self.write("t.json", json.dumps(table))
        out = os.path.join(self.tmp, "selected.txt")
        summary = os.path.join(self.tmp, "summary.md")
        rc = sel.main([
            "--timings", timings_path, "--week", "2026-W36", "--budget-seconds", "60", "--jobs",
            "4", "--output", out, "--summary", summary
        ])
        self.assertEqual(rc, 0)
        selected = sel.read_lines(out)
        self.assertTrue(selected)
        self.assertTrue(set(selected) <= set(names))
        with open(summary, encoding="utf-8") as fh:
            self.assertIn("Fast-lane selection", fh.read())

    def test_comments_and_blanks_are_ignored(self):
        path = self.write("l.txt", "# header\n\nregression/a/x\n  regression/a/y  # trailing\n")
        self.assertEqual(sel.read_lines(path), ["regression/a/x", "regression/a/y"])

    def test_empty_universe_is_an_error(self):
        timings_path = self.write("t.json", json.dumps({"schema": 1, "tests": {}}))
        rc = sel.main(["--timings", timings_path, "--output", os.path.join(self.tmp, "o.txt")])
        self.assertEqual(rc, 1)

    def test_a_test_with_no_measurement_is_still_in_the_universe(self):
        # Without --tests the universe defaults to the timing table's own
        # keys, so an unmeasured test (newly added, or a first-time skip)
        # would never even be considered for the weekly sample -- not merely
        # under-costed. --always-run forces it in deterministically, which
        # requires impute_costs to have given it a real cost in the first
        # place (select() only takes an --always-run name when it has one).
        known = ["regression/suite0/a", "regression/suite0/b"]
        table = {
            "schema": 1,
            "tests": {n: {"seconds": 1.0, "status": "run"} for n in known},
        }
        timings_path = self.write("t.json", json.dumps(table))
        unmeasured = "regression/suite0/brand_new"
        tests_path = self.write("tests.txt", "\n".join(known + [unmeasured]))
        always_run_path = self.write("always.txt", unmeasured)
        out = os.path.join(self.tmp, "selected.txt")
        rc = sel.main([
            "--timings", timings_path, "--tests", tests_path, "--always-run", always_run_path,
            "--week", "2026-W36", "--budget-seconds", "10", "--jobs", "1", "--output", out
        ])
        self.assertEqual(rc, 0)
        self.assertIn(unmeasured, sel.read_lines(out))


class IndexResolution(unittest.TestCase):

    def test_names_map_to_one_based_ctest_numbers(self):
        names = ["alpha", "beta", "gamma"]
        numbers, missing = runner.resolve(["gamma", "alpha"], names)
        self.assertEqual(numbers, [1, 3])
        self.assertEqual(missing, [])

    def test_tests_absent_from_the_build_are_reported_not_dropped_silently(self):
        numbers, missing = runner.resolve(["alpha", "nope"], ["alpha", "beta"])
        self.assertEqual(numbers, [1])
        self.assertEqual(missing, ["nope"])

    def test_whole_suite_stays_inside_the_argument_limit(self):
        # The reason -I is used instead of -R at all: a name alternation over
        # this many tests is past MAX_ARG_STRLEN, an index list is not.
        numbers = list(range(1, 15001))
        spec = "0,0,0," + ",".join(str(n) for n in numbers)
        self.assertLess(len(spec), runner.MAX_ARG_BYTES)


class GreedyCover(unittest.TestCase):

    def test_prefers_the_best_coverage_per_second(self):
        bitsets = {"cheap": 0b1111, "dear": 0b11111111}
        chosen, _ = ccs.greedy_cover(bitsets, {"cheap": 1.0, "dear": 10.0}, budget=100.0)
        self.assertEqual(chosen[0], "cheap")

    def test_covers_everything_reachable_when_the_budget_allows(self):
        bitsets = {"a": 0b0011, "b": 0b1100, "c": 0b0110}
        chosen, trail = ccs.greedy_cover(bitsets, {n: 1.0 for n in bitsets}, budget=100.0)
        self.assertEqual(trail[-1], 1.0)
        self.assertLessEqual(len(chosen), 3)

    def test_skips_redundant_tests(self):
        bitsets = {"a": 0b1111, "b": 0b0011, "c": 0b0001}
        chosen, _ = ccs.greedy_cover(bitsets, {n: 1.0 for n in bitsets}, budget=100.0)
        self.assertEqual(chosen, ["a"])

    def test_respects_the_budget(self):
        bitsets = {f"t{i}": 1 << i for i in range(20)}
        costs = {n: 3.0 for n in bitsets}
        chosen, _ = ccs.greedy_cover(bitsets, costs, budget=10.0)
        self.assertEqual(len(chosen), 3)

    def test_stops_at_the_coverage_target(self):
        bitsets = {f"t{i}": 1 << i for i in range(100)}
        chosen, trail = ccs.greedy_cover(bitsets, {n: 1.0 for n in bitsets}, 1e6, target=0.25)
        self.assertEqual(len(chosen), 25)
        self.assertAlmostEqual(trail[-1], 0.25)

    def test_nothing_fits(self):
        chosen, trail = ccs.greedy_cover({"a": 0b1}, {"a": 50.0}, budget=1.0)
        self.assertEqual((chosen, trail), ([], []))

    def test_no_coverage_at_all(self):
        self.assertEqual(ccs.greedy_cover({"a": 0}, {"a": 1.0}, budget=10.0), ([], []))


class CoverageIo(TempDirCase):

    def test_bitsets_round_trip(self):
        original = {"regression/a/x": (1 << 200) | 0b101, "regression/b/y": 0b11}
        path = os.path.join(self.tmp, ccs.BITSETS)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            for name, bits in original.items():
                raw = bits.to_bytes((bits.bit_length() + 7) // 8, "big")
                blob = base64.b64encode(zlib.compress(raw)).decode("ascii")
                fh.write(json.dumps({"test": name, "bits": blob}) + "\n")
        self.assertEqual(ccs.load_bitsets(path), original)

    def test_profile_directory_names_reverse_to_test_names(self):
        self.assertEqual(ccs.unmangle("regression@esbmc-cpp@cpp@foo"),
                         "regression/esbmc-cpp/cpp/foo")

    def test_select_reports_an_error_with_no_coverage(self):
        os.makedirs(os.path.join(self.tmp, "cov"), exist_ok=True)
        with gzip.open(os.path.join(self.tmp, "cov", ccs.BITSETS), "wt", encoding="utf-8"):
            pass
        rc = ccs.main([
            "select", "--coverage",
            os.path.join(self.tmp, "cov"), "--timings",
            self.write("t.json", json.dumps({"tests": {}})), "--output",
            os.path.join(self.tmp, "core.txt")
        ])
        self.assertEqual(rc, 1)


_TOOLCHAIN = []


def _coverage_toolchain():
    """Find a clang that can actually link an instrumented binary, plus its llvm tools.

    Having clang on PATH is not enough: a packaged clang is routinely installed
    without libclang_rt.profile, and only the link step says so.
    """
    if _TOOLCHAIN:
        return _TOOLCHAIN[0]
    _TOOLCHAIN.append(None)
    for suffix in ("", "-23", "-21", "-20", "-19", "-18", "-17", "-16", "-15", "-14"):
        tools = [shutil.which(f"{t}{suffix}") for t in ("clang", "llvm-cov", "llvm-profdata")]
        if not all(tools):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "probe.c")
            with open(source, "w", encoding="utf-8") as fh:
                fh.write("int main(void) { return 0; }\n")
            probe = subprocess.run([
                tools[0], "-fprofile-instr-generate", "-fcoverage-mapping", "-o",
                os.path.join(tmp, "probe"), source
            ],
                                   capture_output=True,
                                   check=False)
        if probe.returncode == 0:
            _TOOLCHAIN[0] = tools
            return tools
    return None


@unittest.skipUnless(_coverage_toolchain(), "needs clang + llvm-cov + llvm-profdata")
class CoverageCollection(TempDirCase):
    """End-to-end over real llvm-cov output, from raw profiles to a core set."""

    PROGRAM = """#include <stdlib.h>
int a(void) { return 1; }
int b(void) { return 2; }
int c(void) { return 3; }
int main(int argc, char **argv) {
  int which = argc > 1 ? atoi(argv[1]) : 0;
  if (which == 0) return a();
  if (which == 1) return b();
  return c();
}
"""

    def setUp(self):
        super().setUp()
        self.clang, self.llvm_cov, self.llvm_profdata = _coverage_toolchain()
        source = self.write("prog.c", self.PROGRAM)
        self.binary = os.path.join(self.tmp, "prog")
        build = subprocess.run(
            [self.clang, "-fprofile-instr-generate", "-fcoverage-mapping", "-o", self.binary,
             source],
            capture_output=True,
            text=True,
            check=False)
        if build.returncode:
            self.skipTest(f"no usable profile runtime: {build.stderr.strip()[:120]}")

        # Each "test" exercises a different branch, so their line sets differ.
        self.profiles = os.path.join(self.tmp, "profiles")
        for i in range(3):
            env = dict(os.environ)
            # %m, matching what ENABLE_PER_TEST_COVERAGE sets.
            env["LLVM_PROFILE_FILE"] = os.path.join(self.profiles, f"regression@toy@case{i}",
                                                    "%m.profraw")
            subprocess.run([self.binary, str(i)], env=env, check=False)

    def test_collect_then_select(self):
        out = os.path.join(self.tmp, "cov")
        rc = ccs.main([
            "collect", "--profiles", self.profiles, "--binary", self.binary, "--output", out,
            "--jobs", "1", "--exclude", "/nonexistent/", "--llvm-cov", self.llvm_cov,
            "--llvm-profdata", self.llvm_profdata
        ])
        self.assertEqual(rc, 0)

        bitsets = ccs.load_bitsets(os.path.join(out, ccs.BITSETS))
        self.assertEqual(set(bitsets), {f"regression/toy/case{i}" for i in range(3)})
        # Distinct branches must produce distinct line sets, or the cover is
        # solving a degenerate problem.
        self.assertEqual(len(set(bitsets.values())), 3)

        core = os.path.join(self.tmp, "core-set.txt")
        timings_path = self.write(
            "t.json",
            json.dumps({"tests": {n: {"seconds": 1.0} for n in bitsets}}))
        rc = ccs.main([
            "select", "--coverage", out, "--timings", timings_path, "--output", core,
            "--budget-seconds", "30", "--jobs", "1"
        ])
        self.assertEqual(rc, 0)
        self.assertEqual(set(sel.read_lines(core)), set(bitsets))

    def test_a_test_that_wrote_no_profile_is_skipped(self):
        os.makedirs(os.path.join(self.profiles, "regression@toy@crashed"))
        out = os.path.join(self.tmp, "cov")
        self.assertEqual(
            ccs.main([
                "collect", "--profiles", self.profiles, "--binary", self.binary, "--output", out,
                "--jobs", "1", "--llvm-cov", self.llvm_cov, "--llvm-profdata", self.llvm_profdata
            ]), 0)
        self.assertNotIn("regression/toy/crashed", ccs.load_bitsets(os.path.join(out, ccs.BITSETS)))


FAILING_JUNIT = """<?xml version="1.0"?>
<testsuite tests="4">
  <testcase name="regression/esbmc/ok" time="1.0" status="run">
    <system-out>fine</system-out>
  </testcase>
  <testcase name="regression/esbmc/broken" time="2.0" status="fail">
    <failure message="boom">t</failure>
    <system-out>%s</system-out>
  </testcase>
  <testcase name="regression/python/flaky" time="1.0" status="fail">
    <failure message="boom">t</failure>
    <system-out>sometimes</system-out>
  </testcase>
  <testcase name="regression/esbmc/skipped" time="0.0" status="notrun">
    <skipped/>
  </testcase>
</testsuite>
""" % "\n".join(f"line {i}" for i in range(100))


class TempRepo(TempDirCase):
    """A real git repository, so the git-facing code is exercised, not simulated."""

    def setUp(self):
        super().setUp()
        self.repo = os.path.join(self.tmp, "repo")
        os.makedirs(self.repo)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "t@example.com")
        self.git("config", "user.name", "Tester")

    def git(self, *args):
        return subprocess.run(["git", "-C", self.repo, *args],
                              check=True,
                              capture_output=True,
                              text=True).stdout

    def commit(self, message, marker="clean"):
        # marker.txt is what the bisect predicate reads; log.txt only guarantees
        # every commit changes something, so repeating a marker still commits.
        with open(os.path.join(self.repo, "marker.txt"), "w", encoding="utf-8") as handle:
            handle.write(marker + "\n")
        with open(os.path.join(self.repo, "log.txt"), "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD").strip()


class NightlyFailures(TempDirCase):

    def test_only_failing_tests_are_reported(self):
        cases = dict((n, out) for n, _, out in
                     nightly.failing_cases(self.write("j.xml", FAILING_JUNIT)))
        self.assertEqual(set(cases), {"regression/esbmc/broken", "regression/python/flaky"})

    def test_a_skipped_test_is_not_a_failure(self):
        names = [n for n, _, _ in nightly.failing_cases(self.write("j.xml", FAILING_JUNIT))]
        self.assertNotIn("regression/esbmc/skipped", names)

    def test_output_is_truncated_to_the_tail(self):
        cases = dict((n, out) for n, _, out in
                     nightly.failing_cases(self.write("j.xml", FAILING_JUNIT), tail_lines=5))
        tail = cases["regression/esbmc/broken"].splitlines()
        self.assertEqual(len(tail), 5)
        self.assertEqual(tail[-1], "line 99")

    def test_quarantined_failures_are_split_out_and_do_not_drive_the_verdict(self):
        out = os.path.join(self.tmp, "r.json")
        rc = nightly.main([
            "--junit", self.write("j.xml", FAILING_JUNIT), "--commit", "deadbeef",
            "--quarantine", self.write("q.txt", "regression/python/flaky\n"),
            "--json", out
        ])
        self.assertEqual(rc, 0)
        with open(out, encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["quarantined"], ["regression/python/flaky"])
        self.assertEqual([f["test"] for f in report["failures"]], ["regression/esbmc/broken"])


class NightlyOutputs(TempDirCase):

    def test_print_failures_lists_only_unquarantined_names(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            nightly.main([
                "--junit", self.write("j.xml", FAILING_JUNIT), "--commit", "abc",
                "--quarantine", self.write("q.txt", "regression/python/flaky\n"),
                "--print-failures"
            ])
        self.assertEqual(buf.getvalue().split(), ["regression/esbmc/broken"])

    def test_print_failures_prints_nothing_over_the_mass_failure_cutoff(self):
        # De-flaking every failure three times over is exactly the expensive
        # work a mass failure should skip, not run before escalating.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            nightly.main([
                "--junit", self.write("j.xml", FAILING_JUNIT), "--commit", "abc",
                "--max-failures", "1", "--print-failures"
            ])
        self.assertEqual(buf.getvalue(), "")

    def test_github_output_names_one_test_to_bisect(self):
        out = os.path.join(self.tmp, "gh.txt")
        nightly.main([
            "--junit", self.write("j.xml", FAILING_JUNIT), "--commit", "abc",
            "--github-output", out
        ])
        with open(out, encoding="utf-8") as handle:
            written = dict(line.strip().split("=", 1) for line in handle if "=" in line)
        self.assertEqual(written["action"], "escalate")  # no baseline on record
        self.assertEqual(written["test"], "regression/esbmc/broken")

    def test_github_output_on_a_green_run_names_no_test(self):
        green = self.write("g.xml", '<?xml version="1.0"?><testsuite tests="1">'
                           '<testcase name="a" time="1.0" status="run"/></testsuite>')
        out = os.path.join(self.tmp, "gh.txt")
        nightly.main(["--junit", green, "--commit", "abc", "--github-output", out])
        with open(out, encoding="utf-8") as handle:
            written = dict(line.strip().split("=", 1) for line in handle if "=" in line)
        self.assertEqual(written["action"], "none")
        self.assertEqual(written["test"], "")


@unittest.skipUnless(shutil.which("cmake") and shutil.which("ctest"), "needs cmake and ctest")
class Deflaking(TempDirCase):
    """Runs the de-flake pass against a real ctest project with a real flaky test."""

    PROJECT = """cmake_minimum_required(VERSION 3.18)
project(deflake NONE)
enable_testing()
add_test(NAME always_fails COMMAND sh -c "exit 1")
# Passes on exactly the second run: a counter file in the build directory
# survives between ctest invocations, which is what makes it intermittent.
add_test(NAME sometimes_passes COMMAND sh -c
         "n=0; [ -f c ] && n=$(cat c); n=$((n+1)); echo $n > c; test $n -eq 2")
"""

    def setUp(self):
        super().setUp()
        source = os.path.join(self.tmp, "src")
        os.makedirs(source)
        with open(os.path.join(source, "CMakeLists.txt"), "w", encoding="utf-8") as handle:
            handle.write(self.PROJECT)
        self.build = os.path.join(self.tmp, "build")
        done = subprocess.run(["cmake", "-S", source, "-B", self.build],
                              capture_output=True,
                              text=True,
                              check=False)
        if done.returncode:
            self.skipTest(f"cmake configure failed: {done.stderr.strip()[:120]}")

    def test_a_test_that_never_passes_is_a_stable_failure(self):
        stable, passes = bisect.deflake_one(
            "always_fails", 3, lambda t: bisect.run_ctest_test(self.build, t))
        self.assertTrue(stable)
        self.assertEqual(passes, 0)

    def test_an_intermittent_test_is_caught_by_repeating_it(self):
        stable, passes = bisect.deflake_one(
            "sometimes_passes", 3, lambda t: bisect.run_ctest_test(self.build, t))
        self.assertFalse(stable)
        self.assertEqual(passes, 1)

    def test_a_name_that_matches_nothing_is_not_a_pass(self):
        # ctest exits 0 when its filter matches no test, which would otherwise
        # read as the test having passed.
        self.assertFalse(bisect.run_ctest_test(self.build, "no_such_test"))

    def test_flaky_tests_are_quarantined_and_stable_ones_are_not(self):
        quarantine = self.write("q.txt", "# header\nalways_fails\n")
        report = os.path.join(self.tmp, "deflake.json")
        rc = bisect.main([
            "deflake", "--test", "always_fails", "--test", "sometimes_passes", "--build-dir",
            self.build, "--attempts", "3", "--quarantine", quarantine, "--json", report
        ])
        self.assertEqual(rc, 0)
        with open(report, encoding="utf-8") as handle:
            result = json.load(handle)
        self.assertEqual(result["stable"], ["always_fails"])
        self.assertEqual(result["flaky"], ["sometimes_passes"])

        entries = sel.read_lines(quarantine)
        self.assertIn("sometimes_passes", entries)
        # already listed by hand, and stable anyway -- must not be duplicated
        self.assertEqual(entries.count("always_fails"), 1)

    def test_deflake_needs_at_least_one_test(self):
        self.assertEqual(bisect.main(["deflake", "--build-dir", self.build]), 1)


class BisectPredicateGuard(TempDirCase):
    """Exercises bisect_predicate.sh's ctest guard with a fake ctest on PATH,
    rather than a real ESBMC build."""

    SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bisect_predicate.sh")

    def _run(self, ctest_script):
        bin_dir = os.path.join(self.tmp, "bin")
        os.makedirs(bin_dir)
        fake_ctest = os.path.join(bin_dir, "ctest")
        with open(fake_ctest, "w", encoding="utf-8") as handle:
            handle.write(ctest_script)
        os.chmod(fake_ctest, 0o755)

        repo = os.path.join(self.tmp, "repo")
        os.makedirs(os.path.join(repo, "scripts"))
        os.makedirs(os.path.join(repo, "build"))
        with open(os.path.join(repo, "scripts", "build.sh"), "w", encoding="utf-8") as handle:
            handle.write("#!/bin/sh\nexit 0\n")
        os.chmod(os.path.join(repo, "scripts", "build.sh"), 0o755)

        env = dict(os.environ,
                  BISECT_TEST="regression/esbmc/foo",
                  PATH=f"{bin_dir}:{os.environ['PATH']}")
        return subprocess.run([self.SCRIPT],
                              cwd=repo,
                              env=env,
                              capture_output=True,
                              text=True,
                              check=False)

    def test_no_matching_tests_is_skipped_not_good(self):
        # A commit predating the test: ctest matches nothing and exits 0.
        done = self._run('#!/bin/sh\necho "No tests were found!!!" >&2\nexit 0\n')
        self.assertEqual(done.returncode, 125)

    def test_a_real_pass_exits_zero(self):
        done = self._run('#!/bin/sh\necho "1/1 Test #1: foo  Passed"\nexit 0\n')
        self.assertEqual(done.returncode, 0)

    def test_a_real_failure_is_nonzero_and_not_skipped(self):
        done = self._run('#!/bin/sh\necho "1/1 Test #1: foo  Failed"\nexit 8\n')
        self.assertEqual(done.returncode, 8)


class BisectComment(unittest.TestCase):

    def test_a_culprit_comment_names_the_commit_and_disclaims_a_fix(self):
        text = bisect.format_comment({
            "action": "culprit", "commit": "abc1234", "reason": "r",
            "subject": "broke it", "author": "Someone"
        })
        self.assertIn("abc1234", text)
        self.assertIn("broke it", text)
        self.assertIn("human call", text)

    def test_an_escalation_comment_says_why_instead_of_naming_a_commit(self):
        text = bisect.format_comment({"action": "escalate", "reason": "merge commit"})
        self.assertIn("merge commit", text)
        self.assertNotIn("points at", text)


class NightlyVerdict(unittest.TestCase):

    def test_green_needs_no_action(self):
        self.assertEqual(nightly.verdict([], "abc", [])[0], "none")

    def test_a_single_failure_over_a_range_is_bisectable(self):
        self.assertEqual(nightly.verdict(["t"], "abc", [("s", "a", "m")])[0], "bisect")

    def test_a_wall_of_failures_is_an_environment_problem(self):
        action, reason = nightly.verdict([f"t{i}" for i in range(50)], "abc",
                                         [("s", "a", "m")], max_failures=25)
        self.assertEqual(action, "escalate")
        self.assertIn("environment", reason)

    def test_no_baseline_means_nothing_to_bisect(self):
        action, reason = nightly.verdict(["t"], "", [])
        self.assertEqual(action, "escalate")
        self.assertIn("baseline", reason)

    def test_an_unchanged_tree_that_changed_verdict_is_not_deterministic(self):
        action, reason = nightly.verdict(["t"], "abc", [])
        self.assertEqual(action, "escalate")
        self.assertIn("not deterministic", reason)

    def test_an_unreachable_baseline_is_distinguished_from_an_empty_range(self):
        action, reason = nightly.verdict(["t"], "abc", None)
        self.assertEqual(action, "escalate")
        self.assertIn("unreachable", reason)


class NightlyIssueBody(unittest.TestCase):

    def test_an_unreachable_baseline_does_not_crash_the_issue_body(self):
        report = {
            "commit": "abc1234567",
            "last_green": "def4567890",
            "failures": [{"test": "t", "seconds": 1.0, "output": ""}],
            "quarantined": [],
            "commits": None,
            "action": "escalate",
            "reason": "baseline unreachable",
        }
        body = nightly.format_issue(report)
        self.assertIn("commit range unavailable", body)


class NightlyCommitRange(TempRepo):

    def test_lists_commits_since_the_baseline_with_authors(self):
        first = self.commit("one")
        self.commit("two")
        head = self.commit("three")
        commits = nightly.commit_range(self.repo, first, head)
        self.assertEqual([subject for _, _, subject in commits], ["three", "two"])
        self.assertEqual({author for _, author, _ in commits}, {"Tester"})

    def test_an_unreachable_baseline_reports_none_rather_than_crashing(self):
        head = self.commit("one")
        self.assertIsNone(nightly.commit_range(self.repo, "0" * 40, head))


class Deflake(unittest.TestCase):

    def test_a_test_that_never_passes_is_a_stable_failure(self):
        stable, passes = bisect.deflake_one("t", 3, lambda _: False)
        self.assertTrue(stable)
        self.assertEqual(passes, 0)

    def test_one_pass_out_of_three_makes_it_flaky(self):
        # The issue's rule: intermittent tests are quarantined, never bisected.
        results = iter([False, True, False])
        stable, passes = bisect.deflake_one("t", 3, lambda _: next(results))
        self.assertFalse(stable)
        self.assertEqual(passes, 1)


class BisectResult(TempRepo):

    def test_parses_the_culprit_from_git_output(self):
        text = "abc1234567 is the first bad commit\ncommit abc1234567\n"
        self.assertEqual(bisect.parse_first_bad(text), "abc1234567")

    def test_no_culprit_in_the_output(self):
        self.assertEqual(bisect.parse_first_bad("nothing here"), "")

    def test_a_merge_commit_is_detected(self):
        base = self.commit("base")
        self.git("checkout", "-q", "-b", "side")
        self.commit("side work", marker="side")
        self.git("checkout", "-q", "main")
        self.commit("main work", marker="main")
        self.git("merge", "-q", "--no-ff", "-m", "merge", "side", "-X", "ours")
        merge = self.git("rev-parse", "HEAD").strip()
        self.assertTrue(bisect.is_merge_commit(self.repo, merge))
        self.assertFalse(bisect.is_merge_commit(self.repo, base))

    def test_a_merge_commit_is_escalated_not_blamed(self):
        self.commit("base")
        self.git("checkout", "-q", "-b", "side")
        self.commit("side", marker="side")
        self.git("checkout", "-q", "main")
        self.commit("main", marker="main")
        self.git("merge", "-q", "--no-ff", "-m", "merge", "side", "-X", "ours")
        merge = self.git("rev-parse", "HEAD").strip()
        action, reason = bisect.classify(self.repo, merge, True)
        self.assertEqual(action, "escalate")
        self.assertIn("merge commit", reason)

    def test_a_broken_baseline_is_escalated(self):
        sha = self.commit("base")
        action, reason = bisect.classify(self.repo, sha, False)
        self.assertEqual(action, "escalate")
        self.assertIn("baseline is wrong", reason)

    def test_a_plain_commit_is_reported_as_the_culprit(self):
        sha = self.commit("base")
        self.assertEqual(bisect.classify(self.repo, sha, True)[0], "culprit")


class BisectEndToEnd(TempRepo):
    """Drives a real `git bisect run` over a real history."""

    def test_finds_the_commit_that_broke_the_predicate(self):
        good = self.commit("c0")
        for i in range(1, 5):
            self.commit(f"c{i}")
        culprit = self.commit("c5 breaks it", marker="BUG")
        for i in range(6, 9):
            self.commit(f"c{i}", marker="BUG")
        bad = self.git("rev-parse", "HEAD").strip()

        out = os.path.join(self.tmp, "b.json")
        rc = bisect.main([
            "bisect", "--repo", self.repo, "--good", good, "--bad", bad, "--predicate",
            "! grep -q BUG marker.txt", "--verify-good", "--json", out
        ])
        self.assertEqual(rc, 0)
        with open(out, encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["action"], "culprit")
        self.assertEqual(report["commit"], culprit)
        self.assertEqual(report["subject"], "c5 breaks it")

    def test_a_predicate_already_failing_at_good_is_escalated_without_bisecting(self):
        good = self.commit("c0", marker="BUG")
        bad = self.commit("c1", marker="BUG")
        out = os.path.join(self.tmp, "b.json")
        bisect.main([
            "bisect", "--repo", self.repo, "--good", good, "--bad", bad, "--predicate",
            "! grep -q BUG marker.txt", "--verify-good", "--json", out
        ])
        with open(out, encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["action"], "escalate")
        self.assertIn("last known-good", report["reason"])
        self.assertEqual(report["commit"], "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
