#!/usr/bin/env python3

import unittest
from testing_tool import *
from testing_tool import (_add_test, _capped_timeout, _timeout_cap,
                         _TIMEOUT_CAP_ENVVAR)


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
        self.assertEqual(argument_list[-1],
                         os.path.abspath("./esbmc-unix/00_bbuf_02/main.c"))
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
            "./esbmc-unix/00_account_01", "00_account_01")
        self.test_parsed: TestCase = TestCase(
            "./esbmc-unix/00_account_01", "00_account_01")

    def _read_file_checks(self, test_obj: TestCase):
        self.assertEqual(self.test_case.test_mode, "THOROUGH")
        self.assertEqual(self.test_case.test_file, "test.c")
        self.assertEqual(self.test_case.test_args,
                         "account.c --no-slice --context-bound 1 --depth 150")
        self.assertEqual(self.test_case.test_regex, ["^VERIFICATION FAILED$"])

    def _argument_list_checks(self, test_obj: TestCase):
        argument_list = self.test_case.generate_run_argument_list("__test__")
        base = os.path.abspath("./esbmc-unix/00_account_01")
        expected = ['__test__',
                    os.path.join(base, 'account.c'),
                    '--no-slice', '--context-bound', '1', '--depth', '150',
                    os.path.join(base, 'test.c')]
        self.assertEqual(argument_list, expected, str(argument_list))


class CTest4(ParseTest):
    """Added testcase with multiple white spaces in description"""

    def setUp(self):
        self.test_case: TestCase = TestCase(
            "./nonz3/29_exStbHwAcc", "29_exStbHwAcc")
        self.test_parsed: TestCase = TestCase(
            "./nonz3/29_exStbHwAcc", "29_exStbHwAcc")

    def _read_file_checks(self, test_obj: TestCase):
        self.assertEqual(self.test_case.test_mode, "CORE")
        self.assertEqual(self.test_case.test_file, "main.c")
        self.assertEqual(self.test_case.test_args,
                         "--overflow-check  --unwind 3 --32")
        self.assertEqual(self.test_case.test_regex, ["^VERIFICATION FAILED$"])

    def _argument_list_checks(self, test_obj: TestCase):
        argument_list = self.test_case.generate_run_argument_list("__test__")
        expected = ['__test__',
                    '--overflow-check', '--unwind', '3', '--32',
                    os.path.abspath('./nonz3/29_exStbHwAcc/main.c')]
        self.assertEqual(argument_list, expected, str(argument_list))


class ToolTest1(CTest4):
    """Added testcase with multiple white spaces in description"""

    def _argument_list_checks(self, test_obj: TestCase):
        argument_list = self.test_case.generate_run_argument_list(
            "__tool_contains_spaces__ --param 1 __test__")
        expected = ['__tool_contains_spaces__ --param 1 __test__',
                    '--overflow-check', '--unwind', '3', '--32',
                    os.path.abspath('./nonz3/29_exStbHwAcc/main.c')]
        self.assertEqual(argument_list, expected, str(argument_list))


class ToolTest2(CTest4):
    """Added testcase with multiple white spaces in description"""

    def _argument_list_checks(self, test_obj: TestCase):
        argument_list = self.test_case.generate_run_argument_list(
            '__tool_contains_no_spaces__', '--param', '1', '__test__')
        expected = ['__tool_contains_no_spaces__', '--param', '1', '__test__',
                    '--overflow-check', '--unwind', '3', '--32',
                    os.path.abspath('./nonz3/29_exStbHwAcc/main.c')]
        self.assertEqual(argument_list, expected, str(argument_list))


class RelativeTestDirTest(unittest.TestCase):
    """Every test runs ESBMC in a private temporary cwd, so every path the
    runner hands ESBMC must be absolute however the test directory was
    spelled (esbmc/esbmc#4331)."""

    def test_paths_survive_a_chdir(self):
        test_case = TestCase("./nonz3/29_exStbHwAcc", "29_exStbHwAcc")
        source = test_case.generate_run_argument_list("__test__")[-1]
        self.assertTrue(os.path.isabs(source), source)
        with tempfile.TemporaryDirectory() as elsewhere:
            cwd = os.getcwd()
            try:
                os.chdir(elsewhere)
                self.assertTrue(os.path.exists(source), source)
            finally:
                os.chdir(cwd)


class TimeoutReapsProcessGroupTest(unittest.TestCase):
    """A timed-out run must leave nothing behind. ESBMC does not necessarily
    die on SIGTERM -- orphans have been observed still running 34 hours after
    their harness gave up -- so the cleanup has to escalate to SIGKILL."""

    def test_sigterm_ignoring_child_is_reaped(self):
        if os.name != "posix":
            self.skipTest("process-group cleanup is posix-only")
        with tempfile.TemporaryDirectory() as tmp:
            # Stand-in for ESBMC: ignores SIGTERM, then sleeps past the timeout.
            tool = os.path.join(tmp, "ignores_sigterm.py")
            with open(tool, "w", encoding="utf-8") as f:
                f.write(
                    "import os, signal, sys, time\n"
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                    "sys.stderr.write('CHILDPID=' + str(os.getpid()) + '\\n')\n"
                    "sys.stderr.flush()\n"
                    "time.sleep(600)\n"
                )
            test_dir = os.path.join(tmp, "hangs")
            os.mkdir(test_dir)
            open(os.path.join(test_dir, "main.c"), "w").close()
            with open(os.path.join(test_dir, "test.desc"), "w",
                      encoding="utf-8") as f:
                f.write("CORE\nmain.c\n\n^VERIFICATION SUCCESSFUL$\n")

            executor = Executor("{} {}".format(sys.executable, tool))
            executor.timeout = 2
            stdout, stderr, _ = executor.run(TestCase(test_dir, "hangs"))

            self.assertIsNone(stdout, "run should report a timeout")
            marker = re.search(r"CHILDPID=(\d+)", stderr.decode())
            self.assertIsNotNone(marker, stderr.decode())
            child_pid = int(marker.group(1))
            # A reaped pid may be recycled, but not within this test's lifetime.
            with self.assertRaises(OSError):
                os.kill(child_pid, 0)


class CappedTimeoutTest(unittest.TestCase):
    """`ctest --timeout` cannot narrow this suite: CMake gives every test a
    TIMEOUT property, and ctest's flag only defaults tests that lack one. The
    cap env var is the knob that does work (esbmc/esbmc#7628)."""

    def setUp(self):
        self.saved = os.environ.pop(_TIMEOUT_CAP_ENVVAR, None)

    def tearDown(self):
        os.environ.pop(_TIMEOUT_CAP_ENVVAR, None)
        if self.saved is not None:
            os.environ[_TIMEOUT_CAP_ENVVAR] = self.saved

    def test_no_cap_leaves_the_budget_alone(self):
        self.assertEqual(_capped_timeout(1200), 1200)
        self.assertIsNone(_capped_timeout(None))

    def test_a_tighter_cap_wins(self):
        os.environ[_TIMEOUT_CAP_ENVVAR] = "45"
        self.assertEqual(_capped_timeout(1200), 45)

    def test_a_looser_cap_does_not_widen_the_budget(self):
        os.environ[_TIMEOUT_CAP_ENVVAR] = "45"
        self.assertEqual(_capped_timeout(30), 30)

    def test_a_cap_applies_when_there_is_no_budget(self):
        os.environ[_TIMEOUT_CAP_ENVVAR] = "45"
        self.assertEqual(_capped_timeout(None), 45)


def _run_slow_suite(extra_env, desc_requires="", extra_args=()):
    """Run testing_tool.py over a one-test suite whose tool takes 3 seconds."""
    with tempfile.TemporaryDirectory() as tmp:
        tool = os.path.join(tmp, "slow.py")
        with open(tool, "w", encoding="utf-8") as f:
            f.write("import time\n"
                    "time.sleep(3)\n"
                    "print('VERIFICATION SUCCESSFUL')\n")
        test_dir = os.path.join(tmp, "slow")
        os.mkdir(test_dir)
        open(os.path.join(test_dir, "main.c"), "w").close()
        with open(os.path.join(test_dir, "test.desc"), "w",
                  encoding="utf-8") as f:
            f.write("CORE\nmain.c\n\n" + desc_requires +
                    "^VERIFICATION SUCCESSFUL$\n")

        env = dict(os.environ, ESBMC_REGRESS_TIMEOUT="600")
        env.pop(_TIMEOUT_CAP_ENVVAR, None)
        env.update(extra_env)
        return subprocess.run(
            [sys.executable, "testing_tool.py",
             "--tool={} {}".format(sys.executable, tool),
             "--regression=" + tmp, "--modes", "CORE", "--file=slow",
             *extra_args],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env, stdout=PIPE, stderr=PIPE)


class NarrowedBudgetFailsASlowTestTest(unittest.TestCase):
    """The point of the cap: a test that passes on the configured budget must
    fail once the budget is narrowed below its runtime. Without this the suite
    reports a nine-minute run as `Passed` (esbmc/esbmc#7628)."""

    def test_the_slow_test_passes_on_the_configured_budget(self):
        done = _run_slow_suite({})
        self.assertEqual(done.returncode, 0,
                         done.stdout.decode() + done.stderr.decode())

    def test_the_same_test_fails_once_the_budget_is_narrowed(self):
        done = _run_slow_suite({_TIMEOUT_CAP_ENVVAR: "1"})
        output = done.stdout.decode() + done.stderr.decode()
        self.assertNotEqual(done.returncode, 0, output)
        self.assertIn("TIMEOUT TEST", output)
        self.assertIn("capped by " + _TIMEOUT_CAP_ENVVAR, output)


class NarrowedBudgetWithdrawsLongTimeoutTest(unittest.TestCase):
    """`REQUIRES long_timeout` is granted by CMake from the unnarrowed budget,
    so a narrowed run still receives it on the command line. Keeping it fails
    the very tests the capability exists to skip."""

    REQUIRES = "REQUIRES long_timeout\n"
    CAPABILITIES = ("--capabilities=long_timeout", )
    SKIPPED = 10

    def test_the_capability_holds_on_the_configured_budget(self):
        done = _run_slow_suite({}, self.REQUIRES, self.CAPABILITIES)
        self.assertEqual(done.returncode, 0,
                         done.stdout.decode() + done.stderr.decode())

    def test_a_cap_under_ten_minutes_withdraws_it(self):
        done = _run_slow_suite({_TIMEOUT_CAP_ENVVAR: "1"}, self.REQUIRES,
                               self.CAPABILITIES)
        output = done.stdout.decode() + done.stderr.decode()
        self.assertEqual(done.returncode, self.SKIPPED, output)
        self.assertIn("requires long_timeout", output)

    def test_a_cap_of_ten_minutes_or_more_keeps_it(self):
        done = _run_slow_suite({_TIMEOUT_CAP_ENVVAR: "900"}, self.REQUIRES,
                               self.CAPABILITIES)
        self.assertEqual(done.returncode, 0,
                         done.stdout.decode() + done.stderr.decode())


class RejectedTimeoutCapTest(unittest.TestCase):
    """A mis-set cap must stop the run. Ignoring it would leave every test on
    the 1200s budget while the caller believes it was narrowed."""

    def setUp(self):
        self.saved = os.environ.pop(_TIMEOUT_CAP_ENVVAR, None)

    def tearDown(self):
        os.environ.pop(_TIMEOUT_CAP_ENVVAR, None)
        if self.saved is not None:
            os.environ[_TIMEOUT_CAP_ENVVAR] = self.saved

    def test_a_value_that_is_not_a_count_of_seconds_is_refused(self):
        for value in ("45s", "4.5", "-1", "0", "abc"):
            os.environ[_TIMEOUT_CAP_ENVVAR] = value
            with self.assertRaises(SystemExit) as refusal:
                _timeout_cap()
            self.assertIn(_TIMEOUT_CAP_ENVVAR, str(refusal.exception))

    def test_surrounding_whitespace_is_tolerated(self):
        os.environ[_TIMEOUT_CAP_ENVVAR] = " 45 "
        self.assertEqual(_timeout_cap(), 45)


class PrivateCwdTest(unittest.TestCase):
    """A relative output path in the flags line must land in the test's own
    temporary cwd. Letting it resolve against the invocation directory
    overwrote artefacts tracked under regression/ on every run."""

    def _stub(self, tmp):
        """A stand-in for ESBMC that writes the relative file it is given."""
        tool = os.path.join(tmp, "writes_output.py")
        with open(tool, "w", encoding="utf-8") as f:
            f.write("#!" + sys.executable + "\n"
                    "import sys\n"
                    "open(sys.argv[sys.argv.index('--cex-output') + 1], 'w').close()\n"
                    "print('VERIFICATION SUCCESSFUL')\n")
        os.chmod(tool, 0o755)
        test_dir = os.path.join(tmp, "writes")
        os.mkdir(test_dir)
        open(os.path.join(test_dir, "main.c"), "w").close()
        with open(os.path.join(test_dir, "test.desc"), "w", encoding="utf-8") as f:
            f.write("CORE\nmain.c\n--cex-output sideeffect.txt\n"
                    "^VERIFICATION SUCCESSFUL$\n")
        return tool, test_dir

    def _run_from(self, cwd, tool, test_dir):
        previous = os.getcwd()
        try:
            # The runner anchors a relative tool path against the directory it
            # was launched from, so build the executor after the chdir.
            os.chdir(cwd)
            _add_test(TestCase(test_dir, "writes"), Executor(tool))(self)
        finally:
            os.chdir(previous)

    def test_relative_output_stays_out_of_the_invocation_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool, test_dir = self._stub(tmp)
            elsewhere = os.path.join(tmp, "elsewhere")
            os.mkdir(elsewhere)
            self._run_from(elsewhere, tool, test_dir)
            self.assertEqual(os.listdir(elsewhere), [])

    def test_relative_tool_path_resolves_against_the_caller(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool, test_dir = self._stub(tmp)
            relative = os.path.join(os.path.curdir, os.path.basename(tool))
            self._run_from(tmp, relative, test_dir)


if __name__ == '__main__':
    unittest.main()
