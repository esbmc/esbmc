# ci_fixture puts scripts/ci on sys.path, so it comes first.
import ci_fixture  # pylint: disable=wrong-import-order,unused-import
import json
import os
import unittest

import ci_results
import ci_selection
import linux_tests
from ci_results import RuntimeMismatch
from test_bundle import tool_project

HARNESS = ci_results.load_testing_tool()


class PreflightTest(unittest.TestCase):

    def setUp(self):
        self.tmp, self.build = tool_project()
        self.plan = ci_selection.make_plan(self.build)
        self.probes = ci_results.probe_capabilities(self.plan, HARNESS)
        self.paths = {}
        for name, data in (("plan", self.plan), ("probes", self.probes)):
            self.paths[name] = os.path.join(self.tmp.name, f"{name}.json")
            with open(self.paths[name], "w", encoding="utf-8") as fp:
                json.dump(data, fp)

    def tearDown(self):
        self.tmp.cleanup()

    def preflight(self, digest=None, executor="native-1"):
        return linux_tests.main([
            "preflight", "--build-dir", self.build, "--plan", self.paths["plan"], "--plan-digest",
            digest or self.plan["digest"], "--probes", self.paths["probes"], "--executor", executor
        ])

    def test_a_matching_runner_passes(self):
        self.assertTrue(self.probes)
        self.assertEqual(self.preflight(), 0)
        self.assertEqual(self.preflight(executor="python-4"), 0)

    def test_another_producer_plan_is_refused(self):
        self.assertEqual(self.preflight(digest="0" * 64), 1)

    def test_an_edited_plan_is_refused(self):
        self.plan["executors"]["native-1"].append("regression/extra")
        with open(self.paths["plan"], "w", encoding="utf-8") as fp:
            json.dump(self.plan, fp)
        self.assertEqual(self.preflight(), 1)

    def test_a_restored_tree_that_differs_from_the_plan_is_refused(self):
        ctest_file = os.path.join(self.build, "CTestTestfile.cmake")
        with open(ctest_file, encoding="utf-8") as fp:
            text = fp.read()
        with open(ctest_file, "w", encoding="utf-8") as fp:
            fp.write(text.replace("--tool=", "--other-tool="))
        self.assertEqual(self.preflight(), 1)

    def test_a_dynamic_capability_answered_differently_is_refused(self):
        tool = next(iter(self.probes))
        changed = {tool: dict(self.probes[tool], host_cxx_headers=False)}
        with self.assertRaisesRegex(RuntimeMismatch, "dynamic capabilities differ"):
            ci_results.check_runtime(self.plan, changed, HARNESS)


class Z3BinaryTest(unittest.TestCase):

    def test_a_granted_z3_binary_must_be_on_path(self):
        plan = {"tests": {"t": {"command": ["/bin/true", "--capabilities=lp64_host,z3_binary"]}}}
        saved = os.environ["PATH"]
        os.environ["PATH"] = os.path.dirname(os.path.abspath(__file__))
        try:
            with self.assertRaisesRegex(RuntimeMismatch, "no z3"):
                ci_results.check_runtime(plan, {}, HARNESS)
        finally:
            os.environ["PATH"] = saved


if __name__ == "__main__":
    unittest.main()
