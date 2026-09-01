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
import gzip
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
