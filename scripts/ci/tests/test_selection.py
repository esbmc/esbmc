import unittest

import ci_fixture
import ci_selection
from ci_selection import SelectionError


def pr_suite():
    tests = [
        {
            "name": "unit: parses [x]",
            "command": ["/bin/true"]
        },
        {
            "name": "csmith-kind",
            "command": ["/bin/true"]
        },
        ci_fixture.regression("regression/python/intensive_a", "python-intensive/"),
        ci_fixture.regression("regression/esbmc/a[z3]", "esbmc/", extra_labels=["cov-regression"]),
        ci_fixture.regression("regression/floats-regression/b", "floats-regression/"),
        ci_fixture.regression("regression/esbmc-cpp/cpp/c", "esbmc-cpp/cpp/"),
    ]
    for label in ci_selection.PYTHON_LABELS:
        tests.append(ci_fixture.regression(f"regression/{label}t", label))
    return tests


class PlanTest(unittest.TestCase):

    def setUp(self):
        self.tmp, self.build = ci_fixture.project(pr_suite())

    def tearDown(self):
        self.tmp.cleanup()

    def test_executors_partition_the_pr_baseline(self):
        plan = ci_selection.make_plan(self.build)
        names = [n for e in plan["executors"].values() for n in e]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), set(plan["tests"]))
        self.assertNotIn("regression/python/intensive_a", names)
        self.assertEqual(plan["executors"]["producer"], ["unit: parses [x]", "csmith-kind"])
        native = plan["executors"]["native-1"] + plan["executors"]["native-2"]
        self.assertIn("regression/floats-regression/b", native)
        self.assertIn("regression/esbmc/a[z3]", native)
        self.assertEqual(
            sorted(plan["baseline"]),
            sorted(native + [
                n for e in ci_selection.shard_executors() if e.startswith("python")
                for n in plan["executors"][e]
            ]))

    def test_a_missing_shard_breaks_the_partition(self):
        plan = ci_selection.make_plan(self.build)
        parts = dict(plan["executors"])
        del parts["native-2"]
        with self.assertRaisesRegex(SelectionError, "miss"):
            ci_selection.check_partition(plan["tests"], list(parts), parts)

    def test_a_duplicate_hiding_an_omission_is_caught(self):
        plan = ci_selection.make_plan(self.build)
        parts = {e: list(n) for e, n in plan["executors"].items()}
        parts["native-1"].append(parts["native-2"][0])
        parts["native-2"] = parts["native-2"][1:] or [parts["native-1"][0]]
        with self.assertRaisesRegex(SelectionError, "selected by"):
            ci_selection.check_partition(plan["tests"], list(parts), parts)

    def test_two_exclusion_options_are_anded(self):
        # Why the native family needs one union regex: two -LE exclude nothing.
        anded = ci_selection.show_only(self.build, ["-LE", "^numpy/$", "-LE", "^python/$"])
        union = ci_selection.show_only(self.build, ["-LE", "^(numpy/|python/)$"])
        self.assertEqual(len(anded), len(pr_suite()))
        self.assertEqual(len(union), len(pr_suite()) - 2)

    def test_restored_build_must_match_the_plan(self):
        plan = ci_selection.make_plan(self.build)
        ci_selection.check_restored(plan, self.build, "native-1")
        changed = pr_suite()
        changed[3]["command"] = ["/bin/false"]
        ci_fixture.write_project(self.tmp.name, changed)
        ci_fixture.configure(self.tmp.name)
        with self.assertRaisesRegex(SelectionError, "differs"):
            ci_selection.check_restored(plan, self.build, "native-1")


class InventoryTest(unittest.TestCase):

    def plan_of(self, tests):
        tmp, build = ci_fixture.project(tests)
        self.addCleanup(tmp.cleanup)
        return ci_selection.make_plan(build)

    def test_intensive_without_regression_label_is_rejected(self):
        tests = pr_suite()
        tests.append({"name": "stray", "command": ["/bin/true"], "labels": ["python-intensive/"]})
        with self.assertRaisesRegex(SelectionError, "without the regression"):
            self.plan_of(tests)

    def test_scheduling_properties_are_rejected(self):
        tests = pr_suite()
        tests[3]["props"]["RUN_SERIAL"] = "ON"
        with self.assertRaisesRegex(SelectionError, "RUN_SERIAL"):
            self.plan_of(tests)

    def test_a_family_smaller_than_its_shard_count_is_rejected(self):
        # With three Python tests, CTest's `-I 4,,4` does not select nothing:
        # it repeats tests another shard already has.
        dropped = ci_selection.PYTHON_LABELS[3:]
        tests = [t for t in pr_suite() if not t.get("labels") or t["labels"][1] not in dropped]
        with self.assertRaisesRegex(SelectionError, "selected by python-"):
            self.plan_of(tests)

    def test_an_empty_executor_is_rejected(self):
        parts = {"producer": ["a"], "native-1": []}
        with self.assertRaisesRegex(SelectionError, "native-1 selects no test"):
            ci_selection.check_partition(["a"], list(parts), parts)

    def test_a_new_suite_lands_in_the_native_family(self):
        tests = pr_suite() + [ci_fixture.regression("regression/brand-new/x", "brand-new/")]
        plan = self.plan_of(tests)
        native = plan["executors"]["native-1"] + plan["executors"]["native-2"]
        self.assertIn("regression/brand-new/x", native)

    def test_labels_with_regex_characters_are_refused(self):
        with self.assertRaises(SelectionError):
            ci_selection.anchored(["python(/"])


if __name__ == "__main__":
    unittest.main()
