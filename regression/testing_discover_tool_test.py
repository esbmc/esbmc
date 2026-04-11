#!/usr/bin/env python3

from pathlib import Path
import tempfile
import unittest

from testing_discover_tool import discover_tests, generate_ctest_discovery


class DiscoveryTest(unittest.TestCase):
    def _write_test(
        self,
        root_dir: Path,
        relative_dir: str,
        mode: str = "CORE",
        labels: tuple[str, ...] = (),
        test_file: str = "main.c",
        test_args: str = "",
        test_regex: str = "^VERIFICATION SUCCESSFUL$",
    ):
        test_dir = root_dir / relative_dir
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / test_file).write_text(
            "int main(void) { return 0; }\n", encoding="utf-8"
        )
        mode_line = ",".join((mode, *labels))
        (test_dir / "test.desc").write_text(
            f"{mode_line}\n{test_file}\n{test_args}\n{test_regex}\n",
            encoding="utf-8",
        )

    def test_recursive_discovery_sorted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            self._write_test(root_dir, "suite_a/keep_a")
            self._write_test(root_dir, "suite_a/ignored/deeper")
            self._write_test(root_dir, "suite_b/keep_b")
            self._write_test(root_dir, "excluded/skip_me")

            tests = discover_tests(
                root_dir,
                include_prefixes=["suite_b", "suite_a"],
                ignore_prefixes=["suite_a/ignored", "excluded"],
            )

            self.assertEqual(
                [test.relative_dir for test in tests],
                [Path("suite_b/keep_b"), Path("suite_a/keep_a")],
            )
            self.assertEqual(
                [test.labels for test in tests],
                [(), ()],
            )

    def test_recursive_discovery_does_not_normalize_backslash_prefixes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            self._write_test(root_dir, "suite_a/subsuite/keep_me")
            self._write_test(root_dir, "suite_a/ignored/skip_me")

            tests = discover_tests(
                root_dir,
                include_prefixes=["suite_a\\subsuite"],
                ignore_prefixes=["suite_a\\ignored"],
            )

            self.assertEqual(
                [test.relative_dir for test in tests],
                [],
            )

    def test_generate_ctest_discovery_includes_explicit_labels(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            self._write_test(
                root_dir,
                "suite_a/labeled/case_one",
                labels=("nightly", "frontend"),
            )

            tests = discover_tests(
                root_dir,
                include_prefixes=["suite_a"],
                ignore_prefixes=[],
            )

            self.assertEqual(
                [test.labels for test in tests],
                [("nightly", "frontend")],
            )

            output = generate_ctest_discovery(
                root_dir=str(root_dir),
                runner="/path/to/testing_tool.py",
                python_executable="/usr/bin/python3",
                tool="/path/to/esbmc",
                modes=["CORE"],
                memory_limit=64,
                include_prefixes=["suite_a"],
                ignore_prefixes=[],
            )

            self.assertIn(
                '"LABELS" "nightly;frontend;regression;suite_a/labeled/"',
                output,
            )

    def test_generate_ctest_discovery_emits_ctest_runtime_format(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            self._write_test(root_dir, "suite_a/subsuite/case_one")
            self._write_test(root_dir, "suite_a/ignored/case_two")

            output = generate_ctest_discovery(
                root_dir=str(root_dir),
                runner="/path/to/testing_tool.py",
                python_executable="/usr/bin/python3",
                tool="/path/to/esbmc",
                modes=["CORE", "KNOWNBUG"],
                timeout=17,
                memory_limit=64,
                include_prefixes=["suite_a/subsuite"],
                ignore_prefixes=["suite_a/ignored"],
            )

            self.assertIn('add_test("regression/suite_a/subsuite/case_one"', output)
            self.assertNotIn("add_test(NAME", output)
            self.assertIn('"/path/to/testing_tool.py"', output)
            self.assertIn('"--file=case_one"', output)
            self.assertIn(f'"--regression={root_dir}/suite_a/subsuite"', output)
            self.assertIn('"--timeout=17"', output)
            self.assertIn('"--memory-limit=64"', output)
            self.assertIn('"TIMEOUT" "17"', output)
            self.assertNotIn('"ENVIRONMENT"', output)
            self.assertIn('"LABELS" "regression;suite_a/subsuite/"', output)
            self.assertNotIn("suite_a/ignored/case_two", output)


if __name__ == "__main__":
    unittest.main()
