#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os.path
import os
import signal
import sys
import unittest
from subprocess import Popen, PIPE, STDOUT
import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import time
import shlex
import subprocess
from testing_model import TestDescription, FAIL_MODES, TestMode

#####################
# Testing Tool
#####################

# Summary
# - Dynamically generates unittest tests: https://docs.python.org/3/library/unittest.html
# - Support esbmc test description format
# - Sadly unittest does not provide any multiprocessing out-of-box. In the future we can change to a package that
#   extends unittest and adds multiprocessing e.g. nose, testtools. However, it will be an extra dependency on something
#   that is not maintained by python itself.

# Dependencies (install through pip)
# - unittest-xml-reporting

# Bring up a single benchmark
BENCHMARK_BRINGUP = False


class TestCase:

    """Ignore regex and only check for crashes"""
    RUN_ONLY = False
    """SMT only test"""
    SMT_ONLY = False
    """Options that should be removed"""
    UNSUPPORTED_OPTIONS = ["--timeout", "--memlimit"]


def _prepare_child():
    """preexec_fn: new process group + memory cap."""
    os.setpgrp()
    if RegressionBase.MEMORY_LIMIT:
        import resource
        limit = RegressionBase.MEMORY_LIMIT
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ValueError, OSError):
            pass  # macOS does not support RLIMIT_AS
# Seconds to wait between SIGTERM and SIGKILL
# when cleaning up timed-out process group.
_TERM_GRACE = 3

class Executor:
    def __init__(self, tool="esbmc"):
        self.tool = shlex.split(tool)
        self.timeout = RegressionBase.TIMEOUT

    def run(self, test_case: TestDescription):
        """Execute the test case with `executable`"""
        cmd = test_case.generate_run_argument_list(*self.tool, smt_only=TestCase.SMT_ONLY, unsupported_options=TestCase.UNSUPPORTED_OPTIONS)
        preexec = _prepare_child if os.name == "posix" else None

        with subprocess.Popen(
            cmd,
            stdout=PIPE,
            stderr=PIPE,
            preexec_fn=preexec,
            env=dict(os.environ, ESBMC_CONFIG_FILE=""),
        ) as proc:
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                # Gracefully shut down the whole process group so
                # grandchildren don't linger and starve the CI runner.
                if os.name == "posix":
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait(timeout=_TERM_GRACE)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
                stdout, stderr = proc.communicate()
                msg = "Timed out ({}s limit)\nCommand: {}".format(
                    self.timeout, " ".join(str(a) for a in cmd))
                partial = b""
                if stdout:
                    partial += stdout
                if stderr:
                    partial += stderr
                return None, msg.encode() + b"\n" + partial, 1

            if sys.platform.startswith("linux"):
                import resource
                rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                print("mem_usage={0} kilobytes".format(rss))
            return stdout, stderr, proc.returncode


def get_test_objects(base_dir: str) -> list[TestDescription]:
    """Generates a test description from a list of files"""
    assert os.path.exists(base_dir), f"Base directory does not exist: {base_dir}"
    base_path = Path(base_dir).absolute()
    listdir = os.listdir(base_path)
    directories = [x for x in listdir if (base_path / x).is_dir()]
    tests = [
        TestDescription.parse_test_description(base_path / x, base_path.parent)
        for x in directories
    ]
    return tests


class RegressionBase(unittest.TestCase):
    """Base class to use for test generation"""

    longMessage = True

    FAIL_WITH_WORD: str = None
    # Timeout in seconds
    TIMEOUT: int | None = None
    # Memory limit in megabytes
    MEMORY_LIMIT: int | None = None

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        elapsed = time.time() - self.startTime
        if RegressionBase.TIMEOUT and elapsed >= RegressionBase.TIMEOUT:
            print("TIMEOUT after %.1fs (limit %ss)" % (elapsed, RegressionBase.TIMEOUT))
        else:
            print("%.3fs" % elapsed)


def _add_test(test_case : TestDescription, executor: Executor):
    """This method returns a function that defines a test"""

    def test(self):
        stdout, stderr, rc = executor.run(test_case)

        if stdout is None:
            timeout_message = "\nTIMEOUT TEST: {} (limit {}s)".format(
                test_case.test_dir, executor.timeout or "none")
            if stderr:
                timeout_message += "\n" + stderr.decode(errors="replace")
            self.fail(timeout_message)

        if TestCase.RUN_ONLY:
            if rc != 0:
                self.fail(f"Wrong output for process. Bombed out with exit code {rc}")
            return

        output_to_validate = stdout.decode(errors="replace") + stderr.decode(errors="replace")
        error_message_prefix = (
            "\nTEST: "
            + str(test_case.test_dir)
            + "\nEXPECTED TO FIND: "
            + str(test_case.test_regex)
            + "\n\nPROGRAM OUTPUT\n"
        )
        error_message = (
            output_to_validate
            + "\n\nARGUMENTS: "
            + str(test_case.generate_run_argument_list(*executor.tool, smt_only=TestCase.SMT_ONLY, unsupported_options=TestCase.UNSUPPORTED_OPTIONS))
        )

        if BENCHMARK_BRINGUP:
            if os.environ.get("LOG_DIR") is None:
                raise RuntimeError("environment variable LOG_DIR is not defined")
            assert os.path.isdir(os.environ["LOG_DIR"])
            destination = os.environ["LOG_DIR"] + "/" + test_case.name
            f = open(destination, "a")
            f.write("ESBMC args: " + test_case.test_args + "\n\n")
            f.write(output_to_validate)
            f.close()

        matches_regex = True
        for regex in test_case.test_regex:
            match_regex = re.compile(regex, re.MULTILINE)
            if not match_regex.search(output_to_validate.replace("\r", "")):
                matches_regex = False

        if (test_case.test_mode in FAIL_MODES) and matches_regex:
            rel_path = os.path.relpath(test_case.test_dir, os.path.dirname(__file__))
            print(
                f"\033[33mERROR: Test '{rel_path}' passed but is marked as KNOWNBUG. Consider reclassifying it as CORE.\033[0m"
            )
            sys.exit(77)
        elif (test_case.test_mode not in FAIL_MODES) and (not matches_regex):
            if RegressionBase.FAIL_WITH_WORD is not None:
                match_regex = re.compile(RegressionBase.FAIL_WITH_WORD, re.MULTILINE)
                if match_regex.search(output_to_validate):
                    test_case.mark_test_as_knownbug(RegressionBase.FAIL_WITH_WORD)
                    self.fail(error_message_prefix + error_message)
            else:
                self.fail(error_message_prefix + error_message)

    return test


def gen_one_test(base_dir: str, test: str, executor_path: str, modes: list[TestMode]):
    executor = Executor(executor_path)
    base_path = Path(base_dir).absolute()
    test_case = TestDescription.parse_test_description(base_path / test, base_path)
    if test_case.test_mode not in modes:
        exit(10)
    test_func = _add_test(test_case, executor)
    setattr(RegressionBase, "test_{0}".format(test_case.name), test_func)


def _arg_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tool", required=False, help="tool executable path + optional args"
    )
    parser.add_argument("--timeout", required=False, type=int, help="timeout value")
    parser.add_argument("--modes", nargs="+", help="a list of modes that are supported")
    parser.add_argument("--regression", required=False, help="regression suite path")
    parser.add_argument(
        "--mode",
        required=False,
        help="tests to be executed [CORE, " "KNOWNBUG, FUTURE, THOROUGH",
    )
    parser.add_argument("--file", required=False, help="specific test to be executed")
    parser.add_argument(
        "--mark_knownbug_with_word",
        required=False,
        help="If test fails with word then mark it as a knownbug",
    )
    parser.add_argument(
        "--benchbringup",
        default=False,
        action="store_true",
        help="Flag to run a specific benchmark and collect logs in Github workflow",
    )

    parser.add_argument(
        "--smt_test",
        default=False,
        action="store_true",
        help="Replaces usual tests with crash check while producing formulas (adds --smt-formula-only).",
    )
    parser.add_argument(
        "--memory-limit",
        required=False,
        type=int,
        help="Per-test virtual memory limit in megabytes",
    )

    main_args = parser.parse_args()
    if main_args.timeout:
        RegressionBase.TIMEOUT = main_args.timeout
    if main_args.memory_limit:
        RegressionBase.MEMORY_LIMIT = main_args.memory_limit
    RegressionBase.FAIL_WITH_WORD = main_args.mark_knownbug_with_word

    regression_path = os.path.join(
        os.path.dirname(os.path.relpath(__file__)), main_args.regression
    )

    global BENCHMARK_BRINGUP
    if main_args.benchbringup:
        BENCHMARK_BRINGUP = True

    if main_args.smt_test:
        print("Checking SMT generation only")
        TestCase.RUN_ONLY = True
        TestCase.SMT_ONLY = True

    gen_one_test(regression_path, main_args.file, main_args.tool, [TestMode.from_string(mode) for mode in main_args.modes])


def main():
    _arg_parsing()
    suite = unittest.TestLoader().loadTestsFromTestCase(RegressionBase)
    # run all test cases
    unittest.main(argv=[sys.argv[0], "-v"])


if __name__ == "__main__":
    main()

# Utilities for REPL. This helps us to do batch changes over all tests.
# Example:
# 1. Invoke python3 in shell
# 2. `import testing_tool`
# 3. `testing_tool.apply_transform_over_tests(testing_tool.print_test)`

# TODO We probably should obtain this from CMake somehow
TEST_SUITES = [
    "bitwuzla",
    "cbmc",
    "cheri-128",
    "cheri-c",
    "csmith",
    "cstd",
    "cuda/COM_sanity_checks",
    "cuda/Supported_long_time",
    "cuda/benchmarks",
    "cvc",
    "esbmc",
    "esbmc-cpp/algorithm",
    "esbmc-cpp/bug_fixes",
    "esbmc-cpp/cbmc",
    "esbmc-cpp/cpp",
    "esbmc-cpp/deque",
    "esbmc-cpp/gcc-template-tests",
    "esbmc-cpp/inheritance_bringup",
    "esbmc-cpp/list",
    "esbmc-cpp/map",
    "esbmc-cpp/unordered_map",
    "esbmc-cpp/multimap",
    "esbmc-cpp/multiset",
    "esbmc-cpp/unordered_set",
    "esbmc-cpp/OM_sanity_checks",
    "esbmc-cpp/polymorphism_bringup",
    "esbmc-cpp/priority_queue",
    "esbmc-cpp/queue",
    "esbmc-cpp/set",
    "esbmc-cpp/stack",
    "esbmc-cpp/stream",
    "esbmc-cpp/string",
    "esbmc-cpp/template",
    "esbmc-cpp/unix",
    "esbmc-cpp/vector",
    "esbmc-cpp/functional",
    "esbmc-cpp/bitset",
    "esbmc-cpp/unwindsetname",
    "esbmc-cpp11/constructors",
    "esbmc-cpp11/cpp",
    "esbmc-cpp11/new-delete",
    "esbmc-cpp11/reference",
    "esbmc-old",
    "esbmc-solidity",
    "esbmc-unix",
    "esbmc-unix2",
    "extensions",
    "floats",
    "floats-regression",
    "incremental-smt",
    "Interval-analysis-ibex-contractor",
    "jimple",
    "k-induction",
    "goto-contractor",
    "k-induction-parallel",
    "termination" "linux",
    "llvm",
    "mathsat",
    "nonz3",
    "python",
    "smtlib",
    "z3",
    "goto-coverage",
]


def apply_transform_over_tests(functor):
    # Always double check TEST_SUITE variable!
    script_dir_path = os.path.dirname(os.path.relpath(__file__))
    for base_dir in TEST_SUITES:
        test_cases = get_test_objects(os.path.join(script_dir_path, base_dir))
        for test_case in test_cases:
            functor(test_case)


def print_test(test: TestDescription):
    print(str(test))
