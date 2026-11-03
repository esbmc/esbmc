# ci_fixture puts scripts/ci on sys.path, so it comes first.
import ci_fixture  # pylint: disable=wrong-import-order
import os
import subprocess
import tempfile
import unittest

import ci_results
import ci_selection
from ci_results import FAILED, INVALID, PASSED, SKIPPED

HARNESS = ci_results.load_testing_tool()


def run_ctest(build, junit):
    return subprocess.run(["ctest", "--test-dir", build, "--output-junit", junit],
                          capture_output=True,
                          check=False).returncode


class JunitContractTest(unittest.TestCase):
    """The four cases of decision R2, produced by a real CTest run."""

    def test_each_case_maps_to_one_outcome(self):
        tmp, build = ci_fixture.project([
            {
                "name": "passed",
                "command": ["/bin/true"]
            },
            {
                "name": "skipped",
                "command": ["/bin/sh", "-c", "exit 10"],
                "props": {
                    "SKIP_RETURN_CODE": "10"
                }
            },
            {
                "name": "timed_out",
                "command": ["/bin/sleep", "5"],
                "props": {
                    "TIMEOUT": "0.2"
                }
            },
            {
                "name": "not_run",
                "command": [os.path.join(tempfile.gettempdir(), "no-such-tool")]
            },
        ])
        self.addCleanup(tmp.cleanup)
        junit = os.path.join(tmp.name, "junit.xml")
        self.assertNotEqual(run_ctest(build, junit), 0)
        results = ci_results.read_junit(junit)
        self.assertEqual(results["passed"][0], PASSED)
        self.assertEqual(results["skipped"], (SKIPPED, ci_results.SKIP_MESSAGE))
        self.assertEqual(results["timed_out"], (FAILED, "Timeout"))
        self.assertEqual(results["not_run"][0], INVALID)

    def test_a_repeated_name_is_invalid(self):
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as fp:
            fp.write('<testsuite><testcase name="a" status="run"/>'
                     '<testcase name="a" status="run"/></testsuite>')
        self.addCleanup(os.unlink, fp.name)
        self.assertEqual(ci_results.read_junit(fp.name)["a"][0], INVALID)


class SkipAndRunTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.suite = os.path.join(self.tmp.name, "suite")
        ci_fixture.write_case(self.suite, "thorough", "THOROUGH")
        ci_fixture.write_case(self.suite, "needs_z3", "CORE", requires=["z3_binary"])
        ci_fixture.write_case(self.suite, "needs_headers", "CORE", requires=["host_cxx_headers"])
        ci_fixture.write_case(self.suite, "plain", "CORE")
        ci_fixture.write_case(self.suite, "needs_long", "CORE", requires=["long_timeout"])

    def tearDown(self):
        self.tmp.cleanup()

    def case(self, name, exit_code=10, capabilities="lp64_host"):
        return ci_fixture.regression(
            f"regression/suite/{name}", "suite/",
            ci_fixture.harness_command(exit_code, self.suite, name, capabilities=capabilities))

    def evaluate(self, tests, probes=None):
        project = os.path.join(self.tmp.name, "project")
        os.makedirs(project, exist_ok=True)
        ci_fixture.write_project(project, tests)
        build = ci_fixture.configure(project)
        junit = os.path.join(project, "junit.xml")
        exit_code = run_ctest(build, junit)
        plan = {
            "digest": "d",
            "tests": {
                t["name"]: ci_selection.test_record(t)
                for t in ci_selection.show_only(build, [])
            },
        }
        expected = [t["name"] for t in tests]
        return ci_results.evaluate_run(expected, ci_results.read_junit(junit), exit_code, plan,
                                       probes or {}, HARNESS)

    def test_skips_the_harness_explains_pass(self):
        self.assertEqual(self.evaluate([self.case("thorough"), self.case("needs_z3")]), [])

    def long_timeout_case(self, environment):
        test = self.case("needs_long", capabilities="lp64_host,long_timeout")
        test["props"]["ENVIRONMENT"] = environment
        return test

    def test_a_timeout_cap_explains_a_long_timeout_skip(self):
        # The harness drops long_timeout once ESBMC_REGRESS_TIMEOUT_MAX narrows
        # the 1200s budget below 600s, from the test's environment or the run's.
        long_budget = "ESBMC_REGRESS_TIMEOUT=1200"
        self.assertTrue(self.evaluate([self.long_timeout_case(long_budget)]))
        capped = long_budget + ";ESBMC_REGRESS_TIMEOUT_MAX=45"
        self.assertEqual(self.evaluate([self.long_timeout_case(capped)]), [])
        saved = os.environ.get("ESBMC_REGRESS_TIMEOUT_MAX")
        os.environ["ESBMC_REGRESS_TIMEOUT_MAX"] = "45"
        try:
            self.assertEqual(self.evaluate([self.long_timeout_case(long_budget)]), [])
        finally:
            if saved is None:
                del os.environ["ESBMC_REGRESS_TIMEOUT_MAX"]
            else:
                os.environ["ESBMC_REGRESS_TIMEOUT_MAX"] = saved

    def test_an_unexplained_skip_fails(self):
        problems = self.evaluate([self.case("plain")])
        self.assertTrue(any("no reason" in p for p in problems), problems)

    def test_a_skip_needs_the_capability_to_be_absent(self):
        problems = self.evaluate([self.case("needs_z3", capabilities="z3_binary")])
        self.assertTrue(any("no reason" in p for p in problems), problems)

    def test_dynamic_capabilities_come_from_the_producer_probes(self):
        missing = {"esbmc": {"host_cxx_headers": False}}
        present = {"esbmc": {"host_cxx_headers": True}}
        self.assertEqual(self.evaluate([self.case("needs_headers")], missing), [])
        self.assertTrue(self.evaluate([self.case("needs_headers")], present))

    def test_failure_and_exit_code_are_reported(self):
        problems = self.evaluate([self.case("plain", exit_code=1)])
        self.assertTrue(any("exited" in p for p in problems), problems)
        self.assertTrue(any("failed" in p for p in problems), problems)

    def test_a_missing_result_fails(self):
        problems = ci_results.evaluate_run(["a", "b"], {"a": (PASSED, "")}, 0, {}, {}, HARNESS)
        self.assertEqual(problems, ["1 selected tests have no JUnit result: ['b']"])

    def test_an_unselected_result_fails(self):
        results = {"a": (PASSED, ""), "z": (PASSED, "")}
        problems = ci_results.evaluate_run(["a"], results, 0, {}, {}, HARNESS)
        self.assertEqual(problems, ["1 JUnit results were not selected: ['z']"])


class ReportTest(unittest.TestCase):

    def test_missing_and_truncated_junit_fail(self):
        plan = {"digest": "d", "tests": {}}
        with tempfile.TemporaryDirectory() as tmp:
            absent = ci_results.run_report("native-1", ["a"], os.path.join(tmp, "x.xml"), 0, plan,
                                           {}, HARNESS)
            self.assertEqual(absent["problems"], ["no JUnit file"])
            truncated = os.path.join(tmp, "t.xml")
            with open(truncated, "w", encoding="utf-8") as fp:
                fp.write('<testsuite><testcase name="a" status="run">')
            report = ci_results.run_report("native-1", ["a"], truncated, 0, plan, {}, HARNESS)
            self.assertTrue(report["problems"][0].startswith("incomplete JUnit"))
            self.assertIsNotNone(report["junit_digest"])


if __name__ == "__main__":
    unittest.main()
