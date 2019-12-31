#!/usr/bin/env python3

import os.path
import os
import sys
import unittest
from subprocess import Popen, PIPE
import argparse
import re

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


class TestCase:
    """This class is responsible to:
       (a) parse and validate test descriptions.
       (b) hold functions to manipulate and generate commands with test case"""

    def _initialize_test_case(self):
        """Reads test description and initialize this object"""
        with open(self.test_dir + "/test.desc") as fp:
            # First line - TEST MODE
            self.test_mode = fp.readline().strip()
            assert self.test_mode in SUPPORTED_TEST_MODES, str(self.test_mode) + " is not supported"

            # Second line - Test file
            self.test_file = fp.readline().strip()
            assert os.path.exists(self.test_dir + '/' + self.test_file)

            # Third line - Arguments of executable
            self.test_args = fp.readline().strip()

            # Fourth line - Regex of expected output
            self.test_regex = fp.readline().strip()

    def __init__(self, test_dir: str, name: str):
        assert os.path.exists(test_dir)
        assert os.path.exists(test_dir + "/test.desc")
        self.name = name
        self.test_dir = test_dir
        self._initialize_test_case()

    def generate_run_argument_list(self, executable: str):
        """Generates run command list to be used in Popen"""
        result = [executable]
        for x in self.test_args.split(" "):
            if x != "":
                result.append(x)
        result.append(self.test_file)
        return result


class Executor:
    def __init__(self, tool="esbmc"):
        self.tool = tool

    def run(self, test_case: TestCase):
        """Execute the test case with `executable`"""
        process = Popen(test_case.generate_run_argument_list(self.tool), stdout=PIPE, stderr=PIPE,
                        cwd=test_case.test_dir)
        stdout, stderr = process.communicate()
        return stdout, stderr


def _get_test_objects(base_dir: str):
    """Generates a TestCase from a list of files"""
    assert os.path.exists(base_dir)
    listdir = os.listdir(base_dir)
    directories = [x for x in listdir if os.path.isdir(os.path.join(base_dir, x))]
    assert len(directories) > 0
    tests = [TestCase(base_dir + "/" + x, x) for x in directories]
    assert len(tests) > 0
    return tests


class RegressionBase(unittest.TestCase):
    """Base class to use for test generation"""
    longMessage = True


def _add_test(test_case: TestCase, executor):
    """This method returns a function that defines a test"""

    def test(self):
        stdout, stderr = executor.run(test_case)
        regex = re.compile(test_case.test_regex, re.MULTILINE)
        output_to_validate = stdout.decode() + stderr.decode()
        error_message = output_to_validate + str(test_case.generate_run_argument_list(executor.tool))
        self.assertRegex(output_to_validate, regex, msg=error_message)

    return test


def create_tests(executor_path: str, base_dir: str, mode: str):
    assert mode in SUPPORTED_TEST_MODES, str(mode) + " is not supported"

    executor = Executor(executor_path)

    test_cases = _get_test_objects(base_dir)
    assert len(test_cases) > 0
    for test_case in test_cases:
        if test_case.test_mode == mode or mode == "ALL":
            test_func = _add_test(test_case, executor)
            # Add test case into RegressionBase class
            # FUTURE: Maybe change the class name for better report
            setattr(RegressionBase, 'test_{0}'.format(test_case.name), test_func)


def _arg_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, help="tool executable path")
    parser.add_argument("--regression", required=True, help="regression suite path")
    parser.add_argument("--mode", required=True, help="tests to be executed [CORE, "
                                                      "KNOWNBUG, FUTURE, THOROUGH")
    main_args = parser.parse_args()
    return main_args.tool, main_args.regression, main_args.mode


if __name__ == "__main__":
    tool, regression, mode = _arg_parsing()
    create_tests(tool, regression, mode)
    unittest.main(argv=[sys.argv[0]])
