# ci_fixture puts scripts/ci on sys.path, so it comes first.
import ci_fixture  # pylint: disable=wrong-import-order
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest

import ci_bundle
import ci_metrics
import ci_selection
import linux_tests
from ci_bundle import BundleError
from test_selection import pr_suite


def tool_project(tool_mode="OWNER_READ OWNER_WRITE OWNER_EXECUTE"):
    """pr_suite with regression tests calling a tool built into the tree."""
    tmp = tempfile.TemporaryDirectory()
    build = os.path.join(tmp.name, "build")
    tool = os.path.join(build, "tools", "fake-esbmc")
    tests = pr_suite()
    for test in tests:
        if "regression" in test.get("labels", []):
            test["command"] = [tool, f"--tool={tool}"]
    tests.append({"name": "unit-huge", "command": [os.path.join(build, "unit", "huge")]})
    extra = [
        f'file(WRITE "{tool}" "#!/bin/sh\\nexit 0\\n")',
        f'file(CHMOD "{tool}" PERMISSIONS {tool_mode})',
        f'file(WRITE "{build}/unit/huge" "#!/bin/sh\\nexit 0\\n")',
        f'file(CHMOD "{build}/unit/huge" PERMISSIONS OWNER_READ OWNER_EXECUTE)',
        f'file(WRITE "{build}/discovered.cmake" "add_test([==[catch: found]==] /bin/true)\\n")',
        f'set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "{build}/discovered.cmake")',
    ]
    ci_fixture.write_project(tmp.name, tests, extra)
    ci_fixture.configure(tmp.name)
    commit_all(tmp.name)
    return tmp, build


def commit_all(root):
    """A checkout for the bundle's provenance check."""
    subprocess.run(["git", "init", "-q", root], check=True)
    subprocess.run([
        "git", "-C", root, "-c", "user.email=ci@example.com", "-c", "user.name=ci", "commit", "-q",
        "--allow-empty", "-m", "fixture"
    ],
                   check=True)


def head(root):
    return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                          check=True,
                          capture_output=True,
                          text=True).stdout.strip()


class BundleTest(unittest.TestCase):

    def setUp(self):
        self.tmp, self.build = tool_project()
        self.workspace = self.tmp.name
        self.plan = ci_selection.make_plan(self.build)
        self.tar = os.path.join(tempfile.mkdtemp(), "bundle.tar.zst")
        self.addCleanup(shutil.rmtree, os.path.dirname(self.tar))
        self.sha = ci_bundle.create(self.workspace, self.build, self.plan,
                                    {"checkout_sha": head(self.workspace)}, self.tar)

    def tearDown(self):
        self.tmp.cleanup()

    def members(self):
        return subprocess.run(["tar", "-I", "zstd", "-tf", self.tar],
                              check=True,
                              capture_output=True,
                              text=True).stdout.split()

    def test_bundle_holds_ctest_files_and_tools_but_no_unit_binaries(self):
        members = self.members()
        self.assertIn("build/CTestTestfile.cmake", members)
        self.assertIn("build/discovered.cmake", members)
        self.assertIn("build/tools/fake-esbmc", members)
        self.assertNotIn("build/unit/huge", members)

    def test_a_clean_restore_lists_and_runs_the_same_tests(self):
        shutil.move(self.build, self.build + ".orig")
        meta = ci_bundle.restore(self.tar, self.sha, self.workspace)
        self.assertEqual(meta["plan_digest"], self.plan["digest"])
        for executor in ci_selection.shard_executors():
            ci_selection.check_restored(self.plan, self.build, executor)
        names = [t["name"] for t in ci_selection.show_only(self.build, [])]
        self.assertIn("catch: found", names)
        run = subprocess.run(["ctest", "--test-dir", self.build, "-L", "^regression$"],
                             capture_output=True,
                             check=False)
        self.assertEqual(run.returncode, 0, run.stdout)

    def test_a_corrupted_bundle_is_refused(self):
        with open(self.tar, "r+b") as fp:
            fp.seek(100)
            fp.write(b"\0\0\0")
        with self.assertRaisesRegex(BundleError, "digest"):
            ci_bundle.restore(self.tar, self.sha, self.workspace)

    def test_another_workspace_path_is_refused(self):
        with tempfile.TemporaryDirectory() as other:
            with self.assertRaisesRegex(BundleError, "CTest needs that path"):
                ci_bundle.restore(self.tar, self.sha, other)

    def test_sources_of_another_commit_are_refused(self):
        subprocess.run([
            "git", "-C", self.workspace, "-c", "user.email=ci@example.com", "-c", "user.name=ci",
            "commit", "-q", "--allow-empty", "-m", "later"
        ],
                       check=True)
        with self.assertRaisesRegex(BundleError, "sources here"):
            ci_bundle.restore(self.tar, self.sha, self.workspace)

    def test_member_paths_may_not_escape_the_workspace(self):
        evil = os.path.join(os.path.dirname(self.tar), "evil.tar.zst")
        subprocess.run([
            "tar", "-I", "zstd", "-cf", evil, "--transform", "s,^,../,", "-C", self.workspace,
            "CMakeLists.txt"
        ],
                       check=True)
        with self.assertRaisesRegex(BundleError, "escape"):
            ci_bundle.restore(evil, ci_bundle.ci_results.file_digest(evil), self.workspace)


class PermissionTest(unittest.TestCase):

    def test_a_tool_that_lost_its_execute_bit_is_refused(self):
        tmp, build = tool_project(tool_mode="OWNER_READ OWNER_WRITE")
        self.addCleanup(tmp.cleanup)
        plan = ci_selection.make_plan(build)
        tar = os.path.join(tmp.name, "b.tar.zst")
        sha = ci_bundle.create(tmp.name, build, plan, {"checkout_sha": head(tmp.name)}, tar)
        with self.assertRaisesRegex(BundleError, "not executable"):
            ci_bundle.restore(tar, sha, tmp.name)


class SharedLibraryTest(unittest.TestCase):
    """A tool linked against a library of the build tree, as a dynamic Z3 build is."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.build = os.path.join(root, "build")
        with open(os.path.join(root, "lib.c"), "w", encoding="utf-8") as fp:
            fp.write("int answer(void) { return 0; }\n")
        with open(os.path.join(root, "main.c"), "w", encoding="utf-8") as fp:
            fp.write("int answer(void);\nint main(void) { return answer(); }\n")
        tests = pr_suite()
        for test in tests:
            if "regression" in test.get("labels", []):
                test["command"] = [os.path.join(self.build, "tool")]
        ci_fixture.write_project(root, tests, [
            "enable_language(C)", "add_library(fixture SHARED lib.c)",
            "add_executable(tool main.c)", "target_link_libraries(tool fixture)"
        ])
        ci_fixture.configure(root)
        subprocess.run(["cmake", "--build", self.build], check=True, capture_output=True)
        commit_all(root)
        self.plan = ci_selection.make_plan(self.build)
        self.tar = os.path.join(root, "b.tar.zst")
        self.sha = ci_bundle.create(root, self.build, self.plan, {"checkout_sha": head(root)},
                                    self.tar)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_library_travels_with_the_tool(self):
        members = subprocess.run(["tar", "-I", "zstd", "-tf", self.tar],
                                 check=True,
                                 capture_output=True,
                                 text=True).stdout.split()
        self.assertIn("build/libfixture.so", members)
        shutil.move(self.build, self.build + ".orig")
        ci_bundle.restore(self.tar, self.sha, self.tmp.name)
        run = subprocess.run(["ctest", "--test-dir", self.build, "-L", "^regression$"],
                             capture_output=True,
                             check=False)
        self.assertEqual(run.returncode, 0, run.stdout)

    def test_a_library_missing_on_the_runner_is_refused(self):
        shutil.move(self.build, self.build + ".orig")
        subprocess.run(["tar", "-I", "zstd", "-xf", self.tar, "-C", self.tmp.name], check=True)
        os.unlink(os.path.join(self.build, "libfixture.so"))
        lean = os.path.join(self.tmp.name, "lean.tar.zst")
        subprocess.run(
            ["tar", "-I", "zstd", "-cf", lean, "-C", self.tmp.name, "build", ci_bundle.BUNDLE_DIR],
            check=True)
        shutil.rmtree(self.build)
        with self.assertRaisesRegex(BundleError, "cannot load libfixture.so"):
            ci_bundle.restore(lean, ci_bundle.ci_results.file_digest(lean), self.tmp.name)


class RunCommandTest(unittest.TestCase):

    def setUp(self):
        self.tmp, self.build = tool_project()
        self.plan_path = os.path.join(self.tmp.name, "plan.json")
        self.probes = os.path.join(self.tmp.name, "probes.json")
        with open(self.probes, "w", encoding="utf-8") as fp:
            json.dump({}, fp)
        self.assertEqual(
            linux_tests.main(["plan", "--build-dir", self.build, "--out", self.plan_path]), 0)

    def tearDown(self):
        self.tmp.cleanup()

    def run_executor(self, executor):
        out = os.path.join(self.tmp.name, executor)
        code = linux_tests.main([
            "run", "--build-dir", self.build, "--plan", self.plan_path, "--probes", self.probes,
            "--executor", executor, "--out-dir", out
        ])
        with open(os.path.join(out, "report.json"), encoding="utf-8") as fp:
            return code, json.load(fp)

    def test_a_passing_shard_reports_no_problem(self):
        code, report = self.run_executor("c-and-tools")
        self.assertEqual((code, report["problems"]), (0, []))
        self.assertEqual(report["exit_code"], 0)
        self.assertIn("seconds", report["metrics"])

    def test_a_failing_tool_fails_the_shard(self):
        tool = os.path.join(self.build, "tools", "fake-esbmc")
        with open(tool, "w", encoding="utf-8") as fp:
            fp.write("#!/bin/sh\nexit 1\n")
        os.chmod(tool, stat.S_IRWXU)
        code, report = self.run_executor("python-1")
        self.assertEqual(code, 1)
        self.assertNotEqual(report["exit_code"], 0)
        self.assertTrue(any("failed" in p for p in report["problems"]))


class MeasureTest(unittest.TestCase):

    def test_the_exit_code_is_kept_and_memory_is_sampled_or_explained(self):
        code, record = ci_metrics.measure(["/bin/sh", "-c", "sleep 0.3; exit 3"], interval=0.1)
        self.assertEqual((code, record["exit_code"]), (3, 3))
        if record["cgroup"]:
            self.assertGreaterEqual(record["samples"], 1)
            self.assertIn("oom_kills", record)
        else:
            self.assertIn("unavailable", record)


if __name__ == "__main__":
    unittest.main()
