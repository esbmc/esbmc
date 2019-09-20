import unittest
import testing_tool


class TestCaseCore(unittest.TestCase):

    def setUp(self):
        self.test_case = testing_tool.TestCase("./esbmc/00_bbuf_02", "00_bbuf_02")

    def test_read_file_core(self):
        self.assertEqual(self.test_case.test_mode, "CORE")
        self.assertEqual(self.test_case.test_file, "./esbmc/00_bbuf_02/main.c")
        self.assertEqual(self.test_case.test_args, "--unwind 1 --context-bound 2 --schedule --depth 300")
        self.assertEqual(self.test_case.test_regex, "^VERIFICATION FAILED$")

    def test_generate_run_argument_list(self):
        argument_list = self.test_case.generate_run_argument_list("__test__")
        self.assertEqual(argument_list[0], "__test__")
        self.assertEqual(argument_list[1], "--unwind")
