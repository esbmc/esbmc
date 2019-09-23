import unittest
import testing_tool


class TestCaseCore1(unittest.TestCase):
    """This testcase have an argument list"""

    def setUp(self):
        self.test_case = testing_tool.TestCase("./esbmc/00_bbuf_02", "00_bbuf_02")

    def test_read_file_core(self):
        self.assertEqual(self.test_case.test_mode, "CORE")
        self.assertEqual(self.test_case.test_file, "main.c")
        self.assertEqual(self.test_case.test_args, "--unwind 1 --context-bound 2 --schedule --depth 300")
        self.assertEqual(self.test_case.test_regex, "^VERIFICATION FAILED$")

    def test_generate_run_argument_list(self):
        argument_list = self.test_case.generate_run_argument_list("__test__")
        self.assertEqual(argument_list[0], "__test__")
        self.assertEqual(argument_list[1], "--unwind")


class TestCaseCore2(unittest.TestCase):
    """This testcase doesn't have an argument list"""

    def setUp(self):
        self.test_case = testing_tool.TestCase("./llvm/arr", "arr")

    def test_read_file_core(self):
        self.assertEqual(self.test_case.test_mode, "CORE")
        self.assertEqual(self.test_case.test_file, "main.c")
        self.assertEqual(self.test_case.test_args, "")
        self.assertEqual(self.test_case.test_regex, "^VERIFICATION FAILED$")

    def test_generate_run_argument_list(self):
        argument_list = self.test_case.generate_run_argument_list("__test__")
        # Argument list should be the tool + program file
        self.assertEqual(len(argument_list), 2, str(argument_list))


class TestCaseCore_00_account_02(unittest.TestCase):
    """Added testcase with testfile different of main file"""

    def setUp(self):
        self.test_case = testing_tool.TestCase("./esbmc/00_account_02", "00_account_02")

    def test_read_file_core(self):
        self.assertEqual(self.test_case.test_mode, "CORE")
        self.assertEqual(self.test_case.test_file, "test.c")
        self.assertEqual(self.test_case.test_args, "account.c --no-slice --context-bound 1 --depth 150")
        self.assertEqual(self.test_case.test_regex, "^VERIFICATION FAILED$")

    def test_generate_run_argument_list(self):
        argument_list = self.test_case.generate_run_argument_list("__test__")
        expected = ['__test__', 'account.c', '--no-slice', '--context-bound',
                    '1', '--depth', '150', 'test.c']
        self.assertEqual(argument_list, expected, str(argument_list))



class TestCaseCore_29_exStbHwAcc(unittest.TestCase):
    """Added testcase with multiple white spaces in description"""

    def setUp(self):
        self.test_case = testing_tool.TestCase("./esbmc/29_exStbHwAcc", "29_exStbHwAcc")

    def test_read_file_core(self):
        self.assertEqual(self.test_case.test_mode, "CORE")
        self.assertEqual(self.test_case.test_file, "main.c")
        self.assertEqual(self.test_case.test_args, "--overflow-check  --unwind 3")
        self.assertEqual(self.test_case.test_regex, "^VERIFICATION FAILED$")

    def test_generate_run_argument_list(self):
        argument_list = self.test_case.generate_run_argument_list("__test__")
        expected = ['__test__', '--overflow-check', '--unwind', '3', 'main.c']
        self.assertEqual(argument_list, expected, str(argument_list))
