#!/usr/bin/env python3

import os.path
import os
import sys
import unittest
from subprocess import Popen, PIPE
import argparse
import re
import xml.etree.ElementTree as ET

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
CPP_INCLUDE_DIR: str = "DefaultValue"

class BaseTest:
    """This class is responsible to:
       (a) parse and validate test descriptions.
       (b) hold functions to manipulate and generate commands with test case"""

    def _initialize_test_case(self):
        """Reads test description and initialize this object"""
        raise NotImplementedError

    def generate_run_argument_list(self, executable: str):
        """Generates run command list to be used in Popen"""
        result = [executable]
        for x in self.test_args.split(" "):
            if x != "":
                result.append(x)
        result.append(self.test_file)
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
        self._initialize_test_case()

class CTestCase(BaseTest):
    """This specialization will parse C test descriptions"""

    def __init__(self, test_dir: str, name: str):
        super().__init__(test_dir, name)

    def _initialize_test_case(self):
        with open(os.path.join(self.test_dir, "test.desc")) as fp:
            # First line - TEST MODE
            self.test_mode = fp.readline().strip()
            assert self.test_mode in SUPPORTED_TEST_MODES, str(self.test_mode) + " is not supported"

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

class XMLTestCase(BaseTest):
    """This specialization will parse XML test descriptions"""

    UNSUPPORTED_OPTIONS = ["--timeout", "--memlimit"]
    def __init__(self, test_dir: str, name: str):
        super().__init__(test_dir, name)

    def _initialize_test_case(self):
        root = ET.parse(os.path.join(self.test_dir, "test.desc")).getroot()        
        self.version: str = root[0].text.strip()
        self.module: str = root[1].text.strip()
        self.description: str = root[2].text.strip()
        self.test_file: str = root[3].text.strip()
        self.test_args: str = root[4].text.strip()
        self.test_args: str = root[4].text.strip()
        # TODO: Multiline regex
        self.test_regex = [root[5].text.strip()]
        self.priority = root[6].text.strip()
        self.execution_type = root[7].text.strip()
        self.author = root[8].text.strip()

        try:
            self.test_mode = root[9].text.strip()
        except:
            self.test_mode = "CORE"
        finally:
            assert self.test_mode in SUPPORTED_TEST_MODES, str(self.test_mode) + " is not supported"
        assert os.path.exists(os.path.join(self.test_dir, self.test_file))
        
    def generate_run_argument_list(self, executable: str):        
        result = super().generate_run_argument_list(executable)
        # Some sins were committed into test.desc hack them here
        try:
            index = result.index("~/libraries/")
            if CPP_INCLUDE_DIR is None:
                raise RuntimeError(f'[{self.test_dir}] is requesting CPP libraries folder')
            result[index] = CPP_INCLUDE_DIR    
        except ValueError:
            pass

        for x in self.__class__.UNSUPPORTED_OPTIONS:        
            try:
                index = result.index(x)
                result.pop(index)
                result.pop(index)            
            except ValueError:
                pass

        return result

class TestParser:    

    MODES = {"C_TEST": CTestCase, "XML": XMLTestCase}

    @staticmethod
    def detect_mode_by_header(first_line: str) -> str:
        # Look at the header of a file to determine testmode
        # TODO: This should use a better way to detect if its an XML file
        if first_line == "<?xml version='1.0' encoding='utf-8'?>":
            return "XML"
        elif first_line in SUPPORTED_TEST_MODES:
            return "C_TEST"
        raise ValueError(f'Invalid file header: {first_line}')


    @staticmethod
    def from_file(test_dir: str, name: str) -> BaseTest:
        """Tries to open a file and selects which class to parse the file"""
        file_path = os.path.join(test_dir, "test.desc")
        assert os.path.exists(file_path)
        with open(file_path) as fp:            
            first_line = fp.readline().strip()
            return TestParser.MODES[TestParser.detect_mode_by_header(first_line)](test_dir, name)
            

class Executor:
    def __init__(self, tool="esbmc"):
        self.tool = tool

    def run(self, test_case: BaseTest):
        """Execute the test case with `executable`"""
        process = Popen(test_case.generate_run_argument_list(self.tool), stdout=PIPE, stderr=PIPE,
                        cwd=test_case.test_dir)
        stdout, stderr = process.communicate()
        return stdout, stderr


def get_test_objects(base_dir: str):
    """Generates a TestCase from a list of files"""
    assert os.path.exists(base_dir)
    listdir = os.listdir(base_dir)
    directories = [x for x in listdir if os.path.isdir(os.path.join(base_dir, x))]
    assert len(directories) > 10
    tests = [TestParser.from_file(os.path.join(base_dir, x), x) for x in directories]
    assert len(tests) > 10
    return tests


class RegressionBase(unittest.TestCase):
    """Base class to use for test generation"""
    longMessage = True


def _add_test(test_case, executor):
    """This method returns a function that defines a test"""

    def test(self):
        stdout, stderr = executor.run(test_case)
        output_to_validate = stdout.decode() + stderr.decode()
        error_message_prefix = "\nTEST: " + str(test_case.test_dir) + "\nEXPECTED TO FIND: " + str(test_case.test_regex) + "\n\nPROGRAM OUTPUT\n"
        error_message = output_to_validate + "\n\nARGUMENTS: " + str(test_case.generate_run_argument_list(executor.tool))

        matches_regex = True
        for regex in test_case.test_regex:
            match_regex = re.compile(regex, re.MULTILINE)
            if not match_regex.search(output_to_validate):
                matches_regex = False

        if (test_case.test_mode in FAIL_MODES) and matches_regex:
            self.fail(error_message_prefix + error_message)
        elif (test_case.test_mode not in FAIL_MODES) and (not matches_regex):
            self.fail(error_message_prefix + error_message)
    return test


def create_tests(executor_path: str, base_dir: str, mode: str):
    assert mode in SUPPORTED_TEST_MODES, str(mode) + " is not supported"    
    executor = Executor(executor_path)

    test_cases = get_test_objects(base_dir)
    print(f'Found {len(test_cases)} test cases')
    assert len(test_cases) > 0
    for test_case in test_cases:        
        if test_case.test_mode == mode or mode == "ALL":            
            test_func = _add_test(test_case, executor)
            print(f'{test_case.name}')
            # Add test case into RegressionBase class
            # FUTURE: Maybe change the class name for better report
            setattr(RegressionBase, 'test_{0}'.format(test_case.name), test_func)


def _arg_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, help="tool executable path")
    parser.add_argument("--regression", required=True, help="regression suite path")
    parser.add_argument("--mode", required=True, help="tests to be executed [CORE, "
                                                      "KNOWNBUG, FUTURE, THOROUGH")
    parser.add_argument("--library", required=False, help="Path for CPP Library")        
    main_args = parser.parse_args()

    global CPP_INCLUDE_DIR
    CPP_INCLUDE_DIR = main_args.library
    return main_args.tool, main_args.regression, main_args.mode


if __name__ == "__main__":
    tool, regression, mode = _arg_parsing()
    create_tests(tool, regression, mode)
    unittest.main(argv=[sys.argv[0]])
