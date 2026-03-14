#!/usr/bin/env python3

import os
import signal
import subprocess
import shutil
import tempfile
import textwrap
import time
import unittest
from unittest import mock
from testing_tool import *


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
        self.test_case: TestCase = None
        self.test_parsed: TestCase = None

    def _read_file_checks(self, test_obj: TestCase):
        pass

    def _argument_list_checks(self, test_obj: TestCase):
        pass

    def test_case_generation(self):
        for x in [self.test_case, self.test_parsed]:
            self._read_file_checks(x)
            self._argument_list_checks(x)


class CTest1(ParseTest):
    """This testcase have an argument list"""

    def setUp(self):
        self.test_case: TestCase = TestCase(
            "./esbmc-unix/00_bbuf_02", "00_bbuf_02")
        self.test_parsed: TestCase = TestCase(
            "./esbmc-unix/00_bbuf_02", "00_bbuf_02")

    def _read_file_checks(self, test_obj):
        self.assertEqual(test_obj.test_mode, "THOROUGH")
        self.assertEqual(test_obj.test_file, "main.c")
        self.assertEqual(test_obj.test_args,
                         "--unwind 1 --context-bound 2 --schedule --depth 300 -Wno-error=implicit-function-declaration")
        self.assertEqual(test_obj.test_regex, ["^VERIFICATION FAILED$"])

    def _argument_list_checks(self, test_obj):
        argument_list = self.test_case.generate_run_argument_list("__test__")
        self.assertEqual(argument_list[0], "__test__")
        self.assertEqual(argument_list[-1], "./esbmc-unix/00_bbuf_02/main.c")
        self.assertEqual(argument_list[1], "--unwind")


class CTest2(ParseTest):
    """This testcase doesn't have an argument list"""

    def setUp(self):
        self.test_case: TestCase = TestCase("./llvm/arr", "arr")
        self.test_parsed: TestCase = TestCase("./llvm/arr", "arr")

    def _read_file_checks(self, test_obj: TestCase):
        self.assertEqual(self.test_case.test_mode, "CORE")
        self.assertEqual(self.test_case.test_file, "main.c")
        self.assertEqual(self.test_case.test_args, "")
        self.assertEqual(self.test_case.test_regex, ["^VERIFICATION FAILED$"])

    def _argument_list_checks(self, test_obj: TestCase):
        argument_list = self.test_case.generate_run_argument_list("__test__")
        # Argument list should be the tool + program file
        self.assertEqual(len(argument_list), 2, str(argument_list))


class CTest3(ParseTest):
    """Added testcase with testfile different of main file"""

    def setUp(self):
        self.test_case: TestCase = TestCase(
            "./esbmc-unix/00_account_02", "00_account_02")
        self.test_parsed: TestCase = TestCase(
            "./esbmc-unix/00_account_02", "00_account_02")

    def _read_file_checks(self, test_obj: TestCase):
        self.assertEqual(self.test_case.test_mode, "THOROUGH")
        self.assertEqual(self.test_case.test_file, "test.c")
        self.assertEqual(self.test_case.test_args,
                         "account.c --no-slice --context-bound 1 --depth 150")
        self.assertEqual(self.test_case.test_regex, ["^VERIFICATION FAILED$"])

    def _argument_list_checks(self, test_obj: TestCase):
        argument_list = self.test_case.generate_run_argument_list("__test__")
        expected = ['__test__',
                    './esbmc-unix/00_account_02/account.c',
                    '--no-slice', '--context-bound', '1', '--depth', '150', './esbmc-unix/00_account_02/test.c']
        self.assertEqual(argument_list, expected, str(argument_list))


class CTest4(ParseTest):
    """Added testcase with multiple white spaces in description"""

    def setUp(self):
        self.test_case: TestCase = TestCase(
            "./nonz3/29_exStbHwAcc", "29_exStbHwAcc")
        self.test_parsed: TestCase = TestCase(
            "./nonz3/29_exStbHwAcc", "29_exStbHwAcc")

    def _read_file_checks(self, test_obj: TestCase):
        self.assertEqual(self.test_case.test_mode, "KNOWNBUG")
        self.assertEqual(self.test_case.test_file, "main.c")
        self.assertEqual(self.test_case.test_args,
                         "--overflow-check  --unwind 3 --32")
        self.assertEqual(self.test_case.test_regex, ["^VERIFICATION FAILED$"])

    def _argument_list_checks(self, test_obj: TestCase):
        argument_list = self.test_case.generate_run_argument_list("__test__")
        expected = ['__test__',
                    '--overflow-check', '--unwind', '3', '--32', './nonz3/29_exStbHwAcc/main.c']
        self.assertEqual(argument_list, expected, str(argument_list))


class ToolTest1(CTest4):
    """Added testcase with multiple white spaces in description"""

    def _argument_list_checks(self, test_obj: TestCase):
        argument_list = self.test_case.generate_run_argument_list(
            "__tool_contains_spaces__ --param 1 __test__")
        expected = ['__tool_contains_spaces__ --param 1 __test__',
                    '--overflow-check',
                    '--unwind', '3', '--32', './nonz3/29_exStbHwAcc/main.c',]
        self.assertEqual(argument_list, expected, str(argument_list))


class ToolTest2(CTest4):
    """Added testcase with multiple white spaces in description"""

    def _argument_list_checks(self, test_obj: TestCase):
        argument_list = self.test_case.generate_run_argument_list(
            '__tool_contains_no_spaces__', '--param', '1', '__test__')
        expected = ['__tool_contains_no_spaces__', '--param', '1', '__test__',
                    '--overflow-check',
                    '--unwind', '3', '--32', './nonz3/29_exStbHwAcc/main.c']
        self.assertEqual(argument_list, expected, str(argument_list))


class TimeoutBehaviorTest(unittest.TestCase):
    def _create_temp_case(self, test_args: str, runner_body: str):
        tmpdir = tempfile.TemporaryDirectory()
        test_name = "tcase"
        case_dir = os.path.join(tmpdir.name, test_name)
        os.makedirs(case_dir, exist_ok=True)

        with open(os.path.join(case_dir, "test.desc"), "w") as f:
            f.write("CORE\n")
            f.write("main.c\n")
            f.write(f"{test_args}\n")
            f.write(".*\n")

        with open(os.path.join(case_dir, "main.c"), "w") as f:
            f.write("int main(void){return 0;}\n")

        with open(os.path.join(case_dir, "runner.py"), "w") as f:
            f.write(textwrap.dedent(runner_body))

        return tmpdir, TestCase(case_dir, test_name)

    def test_timeout_respects_test_desc_value(self):
        tmpdir, case = self._create_temp_case(
            "runner.py --timeout 3",
            """
            import time
            time.sleep(2)
            print("done")
            """,
        )
        self.addCleanup(tmpdir.cleanup)

        RegressionBase.TIMEOUT = 1
        executor = Executor("python3")
        stdout, stderr, rc = executor.run(case)

        self.assertIsNotNone(stdout)
        self.assertEqual(rc, 0)
        self.assertIn("done", stdout.decode())

    def test_timeout_kills_process_tree(self):
        tmpdir, case = self._create_temp_case(
            "runner.py",
            """
            import os
            import subprocess
            import sys
            import time

            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            pid_path = os.path.join(os.path.dirname(__file__), "child.pid")
            with open(pid_path, "w") as f:
                f.write(str(child.pid))
            time.sleep(30)
            """,
        )
        self.addCleanup(tmpdir.cleanup)
        pid_file = os.path.join(case.test_dir, "child.pid")
        self.addCleanup(lambda: os.path.exists(pid_file) and os.remove(pid_file))

        RegressionBase.TIMEOUT = 1
        executor = Executor("python3")
        stdout, stderr, rc = executor.run(case)

        self.assertIsNone(stdout)
        self.assertEqual(rc, -1)

        self.assertTrue(os.path.exists(pid_file))
        with open(pid_file) as f:
            child_pid = int(f.read().strip())

        time.sleep(0.2)
        try:
            os.kill(child_pid, 0)
            still_alive = True
        except OSError:
            still_alive = False

        if still_alive:
            os.kill(child_pid, signal.SIGKILL)
        self.assertFalse(still_alive)


class WindowsTimeoutFallbackTest(unittest.TestCase):
    def test_windows_taskkill_fallback_kills_parent_when_still_alive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = os.path.join(tmpdir, "tcase")
            os.makedirs(case_dir, exist_ok=True)
            with open(os.path.join(case_dir, "test.desc"), "w") as f:
                f.write("CORE\n")
                f.write("main.c\n")
                f.write("\n")
                f.write(".*\n")
            with open(os.path.join(case_dir, "main.c"), "w") as f:
                f.write("int main(void){return 0;}\n")

            case = TestCase(case_dir, "tcase")
            executor = Executor("esbmc")
            fake_proc = mock.Mock()
            fake_proc.pid = 4321
            fake_proc.communicate.side_effect = subprocess.TimeoutExpired(
                cmd=["esbmc"], timeout=1
            )
            fake_proc.poll.return_value = None
            fake_proc.stdout = None
            fake_proc.stderr = None

            with mock.patch("testing_tool.sys.platform", "win32"), mock.patch(
                "testing_tool.subprocess.Popen", return_value=fake_proc
            ), mock.patch("testing_tool.subprocess.run") as mock_run:
                stdout, stderr, rc = executor.run(case)

            self.assertIsNone(stdout)
            self.assertIsNone(stderr)
            self.assertEqual(rc, -1)
            mock_run.assert_called_once_with(
                ["taskkill", "/PID", "4321", "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            fake_proc.kill.assert_called_once()
            fake_proc.wait.assert_called_once()

    def test_windows_taskkill_success_does_not_kill_parent_again(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = os.path.join(tmpdir, "tcase")
            os.makedirs(case_dir, exist_ok=True)
            with open(os.path.join(case_dir, "test.desc"), "w") as f:
                f.write("CORE\n")
                f.write("main.c\n")
                f.write("\n")
                f.write(".*\n")
            with open(os.path.join(case_dir, "main.c"), "w") as f:
                f.write("int main(void){return 0;}\n")

            case = TestCase(case_dir, "tcase")
            executor = Executor("esbmc")
            fake_proc = mock.Mock()
            fake_proc.pid = 9876
            fake_proc.communicate.side_effect = subprocess.TimeoutExpired(
                cmd=["esbmc"], timeout=1
            )
            # Simulate process already terminated by taskkill.
            fake_proc.poll.return_value = 0
            fake_proc.stdout = None
            fake_proc.stderr = None

            with mock.patch("testing_tool.sys.platform", "win32"), mock.patch(
                "testing_tool.subprocess.Popen", return_value=fake_proc
            ), mock.patch("testing_tool.subprocess.run") as mock_run:
                stdout, stderr, rc = executor.run(case)

            self.assertIsNone(stdout)
            self.assertIsNone(stderr)
            self.assertEqual(rc, -1)
            mock_run.assert_called_once_with(
                ["taskkill", "/PID", "9876", "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            fake_proc.kill.assert_not_called()
            fake_proc.wait.assert_called_once()


class TestPlIntegrationTest(unittest.TestCase):
    def test_timeout_unavailable_is_emitted_once_per_run(self):
        perl_bin = shutil.which("perl")
        bash_bin = shutil.which("bash")
        self.assertIsNotNone(perl_bin)
        self.assertIsNotNone(bash_bin)

        with tempfile.TemporaryDirectory() as tmpdir:
            t1 = os.path.join(tmpdir, "t1")
            t2 = os.path.join(tmpdir, "t2")
            os.makedirs(t1, exist_ok=True)
            os.makedirs(t2, exist_ok=True)

            for tdir in (t1, t2):
                with open(os.path.join(tdir, "main.c"), "w") as f:
                    f.write("int main(void){return 0;}\n")
                with open(os.path.join(tdir, "test.desc"), "w") as f:
                    f.write("CORE\n")
                    f.write("main.c\n")
                    f.write("\n")
                    f.write("ok\n")

            # Build a PATH that contains bash but not timeout/gtimeout.
            fake_bin = os.path.join(tmpdir, "bin")
            os.makedirs(fake_bin, exist_ok=True)
            os.symlink(bash_bin, os.path.join(fake_bin, "bash"))

            env = dict(os.environ)
            env["TEST_TIMEOUT"] = "1"
            env["PATH"] = fake_bin

            test_pl = os.path.join(os.path.dirname(__file__), "test.pl")
            proc = subprocess.run(
                [perl_bin, test_pl, "-c", "echo ok", "t1", "t2"],
                cwd=tmpdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(
                proc.returncode,
                0,
                msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
            )

            with open(os.path.join(t1, "test.out")) as f:
                out1 = f.read()
            with open(os.path.join(t2, "test.out")) as f:
                out2 = f.read()
            unavailable_count = out1.count("TIMEOUT_UNAVAILABLE=1") + out2.count(
                "TIMEOUT_UNAVAILABLE=1"
            )
            self.assertEqual(unavailable_count, 1)


if __name__ == '__main__':
    unittest.main()
