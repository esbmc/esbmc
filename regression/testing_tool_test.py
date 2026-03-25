#!/usr/bin/env python3

import unittest
from pathlib import Path
from unittest.mock import patch
from testing_tool import RegressionBase, _arg_parsing, get_test_objects
from testing_model import TestDescription, TestMode

REGRESSION_ROOT = Path(".").absolute()


class CTestGeneration(unittest.TestCase):
    """This will parse a directory containing C tests and will check for a min/max"""

    def test_quantity(self):
        minimum = 200
        maximum = 5000
        actual = len(get_test_objects("./esbmc"))
        self.assertGreater(actual, minimum)
        self.assertLess(actual, maximum)


class ParseTest(unittest.TestCase):
    """Base Parse Test"""

    def setUp(self):
        self.test_case: TestDescription = None
        self.test_parsed: TestDescription = None

    def _read_file_checks(self, test_obj: TestDescription):
        pass

    def _argument_list_checks(self, test_obj: TestDescription):
        pass

    def test_case_generation(self):
        for x in [self.test_case, self.test_parsed]:
            self._read_file_checks(x)
            self._argument_list_checks(x)


class CTest1(ParseTest):
    """This testcase have an argument list"""

    def setUp(self):
        test_dir = Path("./esbmc-unix/00_bbuf_02").absolute()
        self.test_case = TestDescription.parse_test_description(test_dir, REGRESSION_ROOT)
        self.test_parsed = TestDescription.parse_test_description(test_dir, REGRESSION_ROOT)

    def _read_file_checks(self, test_obj: TestDescription):
        self.assertEqual(test_obj.test_mode, TestMode.THOROUGH)
        self.assertEqual(test_obj.test_file, "main.c")
        self.assertEqual(test_obj.test_args,
                         "--unwind 1 --context-bound 2 --schedule --depth 300 -Wno-error=implicit-function-declaration")
        self.assertEqual(test_obj.test_regex, ("^VERIFICATION FAILED$",))

    def _argument_list_checks(self, test_obj: TestDescription):
        argument_list = self.test_case.generate_run_argument_list("__test__")
        self.assertEqual(argument_list[0], "__test__")
        self.assertEqual(argument_list[-1], str(Path("esbmc-unix/00_bbuf_02/main.c").absolute()))
        self.assertEqual(argument_list[1], "--unwind")


class CTest2(ParseTest):
    """This testcase doesn't have an argument list"""

    def setUp(self):
        test_dir = Path("./llvm/arr").absolute()
        self.test_case = TestDescription.parse_test_description(test_dir, REGRESSION_ROOT)
        self.test_parsed = TestDescription.parse_test_description(test_dir, REGRESSION_ROOT)

    def _read_file_checks(self, test_obj: TestDescription):
        self.assertEqual(self.test_case.test_mode, TestMode.CORE)
        self.assertEqual(self.test_case.test_file, "main.c")
        self.assertEqual(self.test_case.test_args, "")
        self.assertEqual(self.test_case.test_regex, ("^VERIFICATION FAILED$",))

    def _argument_list_checks(self, test_obj: TestDescription):
        argument_list = self.test_case.generate_run_argument_list("__test__")
        # Argument list should be the tool + program file
        self.assertEqual(len(argument_list), 2, str(argument_list))


class CTest3(ParseTest):
    """Added testcase with testfile different of main file"""

    def setUp(self):
        test_dir = Path("./esbmc-unix/00_account_02").absolute()
        self.test_case = TestDescription.parse_test_description(test_dir, REGRESSION_ROOT)
        self.test_parsed = TestDescription.parse_test_description(test_dir, REGRESSION_ROOT)

    def _read_file_checks(self, test_obj: TestDescription):
        self.assertEqual(self.test_case.test_mode, TestMode.THOROUGH)
        self.assertEqual(self.test_case.test_file, "test.c")
        self.assertEqual(self.test_case.test_args,
                         "account.c --no-slice --context-bound 1 --depth 150")
        self.assertEqual(self.test_case.test_regex, ("^VERIFICATION FAILED$",))

    def _argument_list_checks(self, test_obj: TestDescription):
        argument_list = self.test_case.generate_run_argument_list("__test__")
        expected = ['__test__',
                    str(Path("esbmc-unix/00_account_02/account.c").absolute()),
                    '--no-slice', '--context-bound', '1', '--depth', '150', str(Path("esbmc-unix/00_account_02/test.c").absolute())]
        self.assertEqual(argument_list, expected, str(argument_list))


class CTest4(ParseTest):
    """Added testcase with multiple white spaces in description"""

    def setUp(self):
        test_dir = Path("./nonz3/29_exStbHwAcc").absolute()
        self.test_case = TestDescription.parse_test_description(test_dir, REGRESSION_ROOT)
        self.test_parsed = TestDescription.parse_test_description(test_dir, REGRESSION_ROOT)

    def _read_file_checks(self, test_obj: TestDescription):
        self.assertEqual(self.test_case.test_mode, TestMode.KNOWNBUG)
        self.assertEqual(self.test_case.test_file, "main.c")
        self.assertEqual(self.test_case.test_args,
                         "--overflow-check  --unwind 3 --32")
        self.assertEqual(self.test_case.test_regex, ("^VERIFICATION FAILED$",))

    def _argument_list_checks(self, test_obj: TestDescription):
        argument_list = self.test_case.generate_run_argument_list("__test__")
        expected = ['__test__',
                    '--overflow-check', '--unwind', '3', '--32', str(Path("nonz3/29_exStbHwAcc/main.c").absolute())]
        self.assertEqual(argument_list, expected, str(argument_list))


class ToolTest1(CTest4):
    """Added testcase with multiple white spaces in description"""

    def _argument_list_checks(self, test_obj: TestDescription):
        argument_list = self.test_case.generate_run_argument_list(
            "__tool_contains_spaces__ --param 1 __test__")
        expected = ['__tool_contains_spaces__ --param 1 __test__',
                    '--overflow-check',
                    '--unwind', '3', '--32', str(Path("nonz3/29_exStbHwAcc/main.c").absolute()),]
        self.assertEqual(argument_list, expected, str(argument_list))


class ToolTest2(CTest4):
    """Added testcase with multiple white spaces in description"""

    def _argument_list_checks(self, test_obj: TestDescription):
        argument_list = self.test_case.generate_run_argument_list(
            '__tool_contains_no_spaces__', '--param', '1', '__test__')
        expected = ['__tool_contains_no_spaces__', '--param', '1', '__test__',
                    '--overflow-check',
                    '--unwind', '3', '--32', str(Path("nonz3/29_exStbHwAcc/main.c").absolute()),]
        self.assertEqual(argument_list, expected, str(argument_list))


class ArgumentParsingTest(unittest.TestCase):
    def setUp(self):
        self.old_memory_limit = RegressionBase.MEMORY_LIMIT

    def tearDown(self):
        RegressionBase.MEMORY_LIMIT = self.old_memory_limit

    def test_memory_limit_keeps_megabyte_units(self):
        argv = [
            "testing_tool.py",
            "--tool=esbmc",
            "--regression=llvm",
            "--modes",
            "CORE",
            "--file=arr",
            "--memory-limit=64",
        ]

        with patch("testing_tool.gen_one_test"), patch("testing_tool.sys.argv", argv):
            _arg_parsing()

        self.assertEqual(RegressionBase.MEMORY_LIMIT, 64)


if __name__ == '__main__':
    unittest.main()
