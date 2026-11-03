# ci_fixture puts scripts/ci on sys.path, so it comes first.
import ci_fixture  # pylint: disable=wrong-import-order
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest

import ci_results
import ci_selection
import linux_tests
from test_bundle import tool_project

SHARDS = ci_selection.shard_executors()
COMPARE_JOBS = "validate,build,shards,baseline,cpython-check"


def needs(**overrides):
    jobs = {job: {"result": "success"} for job in COMPARE_JOBS.split(",")}
    jobs.update({job: {"result": result} for job, result in overrides.items()})
    return json.dumps(jobs)


class CompareTest(unittest.TestCase):
    """Reports produced by real CTest runs, laid out as downloaded artifacts."""

    def setUp(self):
        self.tmp, self.build = tool_project()
        self.tool = os.path.join(self.build, "tools", "fake-esbmc")
        self.results = os.path.join(self.tmp.name, "results")
        self.plan_path = os.path.join(self.tmp.name, "plan.json")
        self.probes = os.path.join(self.tmp.name, "probes.json")
        linux_tests.main(["plan", "--build-dir", self.build, "--out", self.plan_path])
        with open(self.plan_path, encoding="utf-8") as fp:
            self.digest = json.load(fp)["digest"]
        with open(self.probes, "w", encoding="utf-8") as fp:
            json.dump({}, fp)
        for executor in ["producer", "baseline"] + SHARDS:
            self.run_executor(executor)

    def tearDown(self):
        self.tmp.cleanup()

    def run_executor(self, executor, attempt=1, bundle="b" * 64):
        out = os.path.join(self.results, f"linux-result-{executor}-{attempt}", executor)
        args = [
            "run", "--build-dir", self.build, "--plan", self.plan_path, "--probes", self.probes,
            "--executor", executor, "--out-dir", out
        ]
        if executor != "producer":
            args += ["--bundle-sha256", bundle]
        linux_tests.main(args)

    def set_tool_exit(self, code):
        with open(self.tool, "w", encoding="utf-8") as fp:
            fp.write(f"#!/bin/sh\nexit {code}\n")
        os.chmod(self.tool, stat.S_IRWXU)

    def compare(self, needs_json=None):
        summary = os.path.join(self.tmp.name, "summary.md")
        code = linux_tests.main([
            "compare", "--results-dir", self.results, "--needs", needs_json or needs(),
            "--required", COMPARE_JOBS, "--plan-digest", self.digest, "--bundle-sha256", "b" * 64,
            "--summary", summary
        ])
        with open(summary, encoding="utf-8") as fp:
            return code, fp.read()

    def test_equal_runs_of_the_same_bundle_pass(self):
        code, text = self.compare()
        self.assertEqual(code, 0, text)
        self.assertIn("passed", text)
        self.assertIn("| native-1 |", text)

    def test_a_status_that_differs_from_the_baseline_fails(self):
        self.set_tool_exit(1)
        self.run_executor("baseline", attempt=2)
        code, text = self.compare()
        self.assertEqual(code, 1)
        self.assertIn("baseline failed, shards passed", text)

    def test_a_missing_shard_report_fails(self):
        shutil.rmtree(os.path.join(self.results, "linux-result-native-2-1"))
        code, text = self.compare()
        self.assertEqual(code, 1)
        self.assertIn("native-2 published no report", text)

    def test_a_report_of_another_bundle_fails(self):
        self.run_executor("python-2", attempt=2, bundle="c" * 64)
        code, text = self.compare()
        self.assertEqual(code, 1)
        self.assertIn("python-2 ran bundle", text)

    def test_a_report_of_another_plan_fails(self):
        with open(self.plan_path, encoding="utf-8") as fp:
            plan = json.load(fp)
        plan["digest"] = "f" * 64
        with open(self.plan_path, "w", encoding="utf-8") as fp:
            json.dump(plan, fp)
        self.run_executor("python-3", attempt=2)
        code, text = self.compare()
        self.assertEqual(code, 1)
        self.assertIn("python-3 ran plan " + "f" * 64, text)

    def test_the_latest_attempt_wins_even_when_it_fails(self):
        self.set_tool_exit(1)
        self.run_executor("native-1", attempt=2)
        code, text = self.compare()
        self.assertEqual(code, 1)
        self.assertIn("native-1: ctest exited", text)

    def test_a_failed_or_cancelled_job_fails(self):
        code, text = self.compare(needs(shards="cancelled"))
        self.assertEqual(code, 1)
        self.assertIn("job shards is cancelled", text)


class ExpectedFailureVerdictTest(unittest.TestCase):
    """A .desc expecting VERIFICATION FAILED is a passing test, through the real harness."""

    def test_the_harness_verdict_is_what_counts(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        suite = os.path.join(tmp.name, "suite")
        ci_fixture.write_case(suite, "fails", "CORE")
        with open(os.path.join(suite, "fails", "test.desc"), "w", encoding="utf-8") as fp:
            fp.write("CORE\nmain.c\n\n^VERIFICATION FAILED$\n")
        tool = os.path.join(tmp.name, "fake-esbmc")
        with open(tool, "w", encoding="utf-8") as fp:
            fp.write("#!/bin/sh\necho 'VERIFICATION FAILED'\nexit 1\n")
        os.chmod(tool, stat.S_IRWXU)
        harness = os.path.join(os.path.dirname(ci_results.__file__), "..", "..", "regression",
                               "testing_tool.py")
        test = ci_fixture.regression("regression/suite/fails", "suite/", [
            sys.executable,
            os.path.abspath(harness), f"--tool={tool}", f"--regression={suite}", "--modes", "CORE",
            "--file=fails"
        ])
        ci_fixture.write_project(tmp.name, [test])
        build = ci_fixture.configure(tmp.name)
        junit = os.path.join(tmp.name, "junit.xml")
        code = ci_fixture.subprocess.run(["ctest", "--test-dir", build, "--output-junit", junit],
                                         capture_output=True,
                                         check=False).returncode
        plan = {"tests": {test["name"]: ci_selection.test_record(test)}}
        self.assertEqual(code, 0)
        self.assertEqual(
            ci_results.evaluate_run([test["name"]], ci_results.read_junit(junit), code, plan, {},
                                    ci_results.load_testing_tool()), [])


if __name__ == "__main__":
    unittest.main()
