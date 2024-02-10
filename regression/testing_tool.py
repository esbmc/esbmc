#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os.path
import os
import sys
import unittest
from subprocess import Popen, PIPE, STDOUT
import argparse
import re
import xml.etree.ElementTree as ET
import time
import shlex
import subprocess

if sys.platform.startswith('linux'):
    from resource import *

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

# TestModes
# CORE -> Essential tests that are fast
# THOROUGH -> Slower tests
# KNOWNBUG -> Tests that are known to fail due to bugs
# FUTURE -> Test that are known to fail due to missing implementation
# ALL -> Run all tests
SUPPORTED_TEST_MODES = ["CORE", "FUTURE", "THOROUGH", "KNOWNBUG", "ALL"]
FAIL_MODES = ["KNOWNBUG"]

# Bring up a single benchmark
BENCHMARK_BRINGUP = False

class TestCase:
    """This class is responsible to:
       (a) parse and validate test descriptions.
       (b) hold functions to manipulate and generate commands with test case"""

    def _initialize_test_case(self):
        """Reads test description and initialize this object"""
        with open(os.path.join(self.test_dir, "test.desc")) as fp:
            # First line - TEST MODE
            self.test_mode = fp.readline().strip()
            assert self.test_mode in SUPPORTED_TEST_MODES, str(
                self.test_mode) + " is not supported"

            # Second line - Test file
            self.test_file = fp.readline().strip()
            assert os.path.exists(self.test_dir + '/' + self.test_file)

            # Third line - Arguments of executable
            self.test_args = fp.readline().strip()

            # Fourth line and beyond
            # Regex of expected output
            self.test_regex = []
            for line in fp:
                self.test_regex.append(line.strip())


    def generate_run_argument_list(self, *tool):
        """Generates run command list to be used in Popen"""
        result = list(tool)
        result.append(os.path.join(self.test_dir, self.test_file))
        for x in shlex.split(self.test_args):
            if x != "":
                p = os.path.join(self.test_dir, x)
                result.append(p if os.path.exists(p) else x)
        if TestCase.SMT_ONLY:
            result.append("--smtlib")
            result.append("--smt-formula-only")
            result.append("--output")
            result.append(f"{self.test_dir}.smt2")
            result.append("--array-flattener")

        for x in TestCase.UNSUPPORTED_OPTIONS:
            try:
                index = result.index(x)
                result.pop(index)
                result.pop(index)
            except ValueError:
                pass    

        return result

    def __str__(self):
        return f'[{self.name}]: {self.test_dir}, {self.test_mode}'

    def __init__(self, test_dir: str, name: str):
        assert os.path.exists(test_dir)
        assert os.path.exists(os.path.join(test_dir, "test.desc"))
        self.name = name
        self.test_dir = test_dir
        self.test_args = None
        self.test_file = None
        self.test_mode = "CORE"
        self._initialize_test_case()

    def save_test(self):
        """Replaces original test with the current configuration"""
        test_desc_path = os.path.join(self.test_dir, "test.desc")
        assert(os.path.isfile(test_desc_path))
        with open(test_desc_path, 'w') as f:
            f.write(f"{self.test_mode}\n")
            f.write(f"{self.test_file}\n")
            f.write(f"{self.test_args}\n")
            for re in self.test_regex:
                f.write(f"{re}\n")


    """Ignore regex and only check for crashes"""
    RUN_ONLY = False
    """SMT only test"""
    SMT_ONLY = False
    """Options that should be removed"""
    UNSUPPORTED_OPTIONS = ["--timeout", "--memlimit"]



class Executor:
    def __init__(self, tool="esbmc"):
        self.tool = shlex.split(tool)
        self.timeout = RegressionBase.TIMEOUT

    def run(self, test_case: TestCase):
        """Execute the test case with `executable`"""
        cmd = test_case.generate_run_argument_list(*self.tool)

        try:
            # use subprocess.run because we want to wait for the subprocess to finish
            p = subprocess.run(cmd, stdout=PIPE, stderr=PIPE, timeout=self.timeout);

            # get the RSS (resident set size) of the subprocess that just terminated.
            # Save the output in a tmp.log and then use the command below
            # to get the total maximum RSS:
            #   egrep "mem_usage=[0-9]+" tmp.log -o | cut -d'=' -f2 | paste -sd+ - | bc
            # see https://docs.python.org/3/library/resource.html for more details
            if sys.platform.startswith('linux'):
                print("mem_usage={0} kilobytes".format(getrusage(RUSAGE_CHILDREN).ru_maxrss))

        except subprocess.CalledProcessError:
            return None, None, 0

        return p.stdout, p.stderr, p.returncode


def get_test_objects(base_dir: str):
    """Generates a TestCase from a list of files"""
    assert os.path.exists(base_dir)
    listdir = os.listdir(base_dir)
    directories = [x for x in listdir if os.path.isdir(
        os.path.join(base_dir, x))]
    tests = [TestCase(os.path.join(base_dir, x), x)
             for x in directories]
    return tests

class RegressionBase(unittest.TestCase):
    """Base class to use for test generation"""
    longMessage = True

    FAIL_WITH_WORD: str = None
    TIMEOUT = None

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        if RegressionBase.TIMEOUT and t >= RegressionBase.TIMEOUT:
            print('TIMEOUT')
        else:
            print('%.3f' %  t)

def _add_test(test_case, executor):
    """This method returns a function that defines a test"""

    def test(self):
        stdout, stderr, rc = executor.run(test_case)

        if stdout == None:
            timeout_message ="\nTIMEOUT TEST: " + str(test_case.test_dir)
            self.fail(timeout_message)

        if TestCase.RUN_ONLY:
            if rc != 0:
                self.fail(f"Wrong output for process. Bombed out with exit code {rc}")
            return

        output_to_validate = stdout.decode() + stderr.decode()
        error_message_prefix = "\nTEST: " + \
            str(test_case.test_dir) + "\nEXPECTED TO FIND: " + \
            str(test_case.test_regex) + "\n\nPROGRAM OUTPUT\n"
        error_message = output_to_validate + "\n\nARGUMENTS: " + \
            str(test_case.generate_run_argument_list(*executor.tool))

        if(BENCHMARK_BRINGUP):
            if os.environ.get('LOG_DIR') is None:
                raise RuntimeError('environment variable LOG_DIR is not defined')
            assert os.path.isdir(os.environ['LOG_DIR'])
            destination = os.environ['LOG_DIR'] + '/' + test_case.name
            f=open(destination, 'a')
            f.write("ESBMC args: " + test_case.test_args + '\n\n')
            f.write(output_to_validate)
            f.close()

        matches_regex = True
        for regex in test_case.test_regex:
            match_regex = re.compile(regex, re.MULTILINE)
            if not match_regex.search(output_to_validate.replace("\r", "")):
                matches_regex = False

        if (test_case.test_mode in FAIL_MODES) and matches_regex:
            self.fail(error_message_prefix + error_message)
        elif (test_case.test_mode not in FAIL_MODES) and (not matches_regex):
            if RegressionBase.FAIL_WITH_WORD is not None:
                match_regex = re.compile(RegressionBase.FAIL_WITH_WORD, re.MULTILINE)
                if match_regex.search(output_to_validate):
                    test_case.mark_test_as_knownbug(RegressionBase.FAIL_WITH_WORD)
                    self.fail(error_message_prefix + error_message)
            else:
                self.fail(error_message_prefix + error_message)
    return test


def gen_one_test(base_dir: str, test: str, executor_path: str, modes):
    executor = Executor(executor_path)
    test_case = TestCase(os.path.join(base_dir, test), test)
    if test_case.test_mode not in modes:
        exit(10)
    test_func = _add_test(test_case, executor)
    setattr(RegressionBase, 'test_{0}'.format(test_case.name), test_func)


def _arg_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=False, help="tool executable path + optional args")
    parser.add_argument("--timeout", required=False, help="timeout value")
    parser.add_argument('--modes', nargs='+', help="a list of modes that are supported")
    parser.add_argument("--regression", required=False,
                        help="regression suite path")
    parser.add_argument("--mode", required=False, help="tests to be executed [CORE, "
                                                      "KNOWNBUG, FUTURE, THOROUGH")
    parser.add_argument("--file", required=False, help="specific test to be executed")
    parser.add_argument("--mark_knownbug_with_word", required=False,
                        help="If test fails with word then mark it as a knownbug")
    parser.add_argument("--benchbringup", default=False, action="store_true",
            help="Flag to run a specific benchmark and collect logs in Github workflow")

    parser.add_argument("--smt_test", default=False, action="store_true",
            help="Replaces usual tests with crash check while producing formulas (adds --smt-formula-only).")

    main_args = parser.parse_args()
    if main_args.timeout:
        RegressionBase.TIMEOUT = int(main_args.timeout)
    RegressionBase.FAIL_WITH_WORD = main_args.mark_knownbug_with_word

    regression_path = os.path.join(os.path.dirname(os.path.relpath(__file__)),
                                   main_args.regression)

    global BENCHMARK_BRINGUP
    if(main_args.benchbringup):
        BENCHMARK_BRINGUP = True

    if(main_args.smt_test):
        print("Checking SMT generation only")
        TestCase.RUN_ONLY = True
        TestCase.SMT_ONLY = True

    gen_one_test(regression_path, main_args.file, main_args.tool, main_args.modes)    

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
    "esbmc-cpp/multimap",
    "esbmc-cpp/multiset",
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
    "k-induction-parallel",
    "linux",
    "llvm",
    "mathsat",
    "nonz3",
    "python",
    "smtlib",
    "z3",
]

def apply_transform_over_tests(functor):
    # Always double check TEST_SUITE variable!
    for base_dir in TEST_SUITES:
        test_cases = get_test_objects(base_dir)
        for test_case in test_cases:
            functor(test_case)

def print_test(test: TestCase):
    print(str(test))

