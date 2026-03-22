#!/usr/bin/env python3

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import os
import sys

SUPPORTED_TEST_MODES: list[str] = ["CORE", "FUTURE", "THOROUGH", "KNOWNBUG", "ALL"]
_TIMEOUT_ENVVAR: str = "ESBMC_REGRESS_TIMEOUT"
_MEMORY_LIMIT_ENVVAR: str = "ESBMC_REGRESS_MEMORY_LIMIT"


def _normalize_relative_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip("/")
    return "" if normalized == "." else normalized


def _relative_dir(root_dir: str, path: str) -> str:
    return _normalize_relative_path(os.path.relpath(path, root_dir))


def _is_same_or_child(path: str, prefix: str) -> bool:
    if not prefix:
        return True
    return path == prefix or path.startswith(prefix + "/")


def _should_include_path(
    path: str, include_prefixes: Sequence[str], ignore_prefixes: Sequence[str]
) -> bool:
    if any(_is_same_or_child(path, prefix) for prefix in ignore_prefixes):
        return False
    if not include_prefixes:
        return True
    return any(_is_same_or_child(path, prefix) for prefix in include_prefixes)


def _should_descend_into(
    path: str, include_prefixes: Sequence[str], ignore_prefixes: Sequence[str]
) -> bool:
    if any(_is_same_or_child(path, prefix) for prefix in ignore_prefixes):
        return False
    if not include_prefixes:
        return True
    return any(
        _is_same_or_child(path, prefix) or _is_same_or_child(prefix, path)
        for prefix in include_prefixes
    )


def _suite_order_for_path(path: str, include_prefixes: Sequence[str]) -> int:
    for index, prefix in enumerate(include_prefixes):
        if _is_same_or_child(path, prefix):
            return index
    return len(include_prefixes)


def _cmake_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _cmake_join(arguments: Sequence[str]) -> str:
    return " ".join(_cmake_quote(argument) for argument in arguments)


@dataclass(frozen=True)
class TestDescription:
    """Immutable subset of test.desc fields needed during CTest discovery."""

    test_dir: str
    relative_dir: str
    test_mode: str
    test_file: str
    labels: list[str]

    @staticmethod
    def parse_test_description(test_dir: str, relative_dir: str) -> "TestDescription":
        """Read the minimal test.desc header needed to register the test."""
        test_desc_path = os.path.join(test_dir, "test.desc")
        with open(test_desc_path, encoding="utf-8") as fp:
            test_mode = fp.readline().rstrip("\r\n")
            # assert (
            #     test_mode.strip() == test_mode
            # ), f"{test_dir}: test mode line must not have leading/trailing whitespace: '{test_mode}'"
            test_mode = test_mode.strip()
            assert (
                test_mode in SUPPORTED_TEST_MODES
            ), f"{test_dir}: {test_mode} is not supported"
            test_file = fp.readline().strip()
            assert os.path.exists(os.path.join(test_dir, test_file))
            fp.readline()
        test_lables = TestDescription.getDefaultLabels(relative_dir)
        return TestDescription(
            test_dir, relative_dir, test_mode, test_file, test_lables
        )

    @staticmethod
    def getDefaultLabels(relative_dir: str) -> list[str]:
        """Return the built-in ctest labels for a discovered regression test."""
        return ["regression", os.path.dirname(relative_dir) + "/"]


def discover_tests(
    root_dir: str,
    include_prefixes: Sequence[str],
    ignore_prefixes: Sequence[str],
) -> list[TestDescription]:
    """Recursively discover test.desc files below root_dir."""
    normalized_include_prefixes: list[str] = [
        _normalize_relative_path(prefix)
        for prefix in (include_prefixes or [])
        if prefix
    ]
    normalized_ignore_prefixes: list[str] = [
        _normalize_relative_path(prefix) for prefix in (ignore_prefixes or []) if prefix
    ]

    root_dir = os.path.abspath(root_dir)
    discovered: list[TestDescription] = []

    for current_dir, dirnames, filenames in os.walk(root_dir, topdown=True):
        dirnames.sort()
        rel_dir = _relative_dir(root_dir, current_dir)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if _should_descend_into(
                _normalize_relative_path(os.path.join(rel_dir, dirname)),
                normalized_include_prefixes,
                normalized_ignore_prefixes,
            )
        ]

        if "test.desc" not in filenames:
            continue
        if not _should_include_path(
            rel_dir, normalized_include_prefixes, normalized_ignore_prefixes
        ):
            continue

        test = TestDescription.parse_test_description(current_dir, rel_dir)
        discovered.append(test)

    discovered.sort(
        key=lambda test: (
            _suite_order_for_path(test.relative_dir, normalized_include_prefixes),
            test.relative_dir,
        )
    )
    return discovered


def generate_ctest_discovery(
    root_dir: str,
    runner: str,
    python_executable: str,
    tool: str,
    modes: Sequence[str],
    include_prefixes: Sequence[str],
    ignore_prefixes: Sequence[str],
    timeout: int | None = None,
    memory_limit: int | None = None,
    benchbringup: bool = False,
) -> str:
    """Render discovered tests as the add_test script consumed by CTest."""
    root_dir = os.path.abspath(root_dir)
    runner = os.path.abspath(runner)
    tests: list[TestDescription] = discover_tests(
        root_dir, include_prefixes, ignore_prefixes
    )

    lines: list[str] = []
    for test in tests:
        test_name = f"regression/{test.relative_dir}"

        command: list[str] = [
            python_executable,
            runner,
            f"--tool={tool}",
            f"--regression={root_dir}",
            "--modes",
            *modes,
            f"--file={test.relative_dir}",
        ]
        if benchbringup:
            command.append("--benchbringup")

        lines.append(f"add_test({_cmake_join([test_name, *command])})")

        properties: list[str] = ["SKIP_RETURN_CODE", "10"]
        if timeout is not None:
            properties.extend(["TIMEOUT", str(timeout)])

        environment: list[str] = []
        if timeout is not None:
            environment.append(f"{_TIMEOUT_ENVVAR}={timeout}")
        if memory_limit is not None:
            environment.append(f"{_MEMORY_LIMIT_ENVVAR}={memory_limit}")
        if environment:
            properties.extend(["ENVIRONMENT", ";".join(environment)])

        labels = ";".join(test.labels)
        if labels:
            properties.extend(["LABELS", labels])

        lines.append(
            f"set_tests_properties({_cmake_quote(test_name)} PROPERTIES {_cmake_join(properties)})"
        )

    return "\n".join(lines) + ("\n" if lines else "")


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover regression tests under a root directory and emit "
        "the CTest add_test() definitions used at ctest runtime."
        "This allows to dynamically discover tests without CMake having to be involved in the discovery process."
    )
    parser.add_argument(
        "--root", required=True, help="root directory to scan recursively"
    )
    parser.add_argument(
        "--runner",
        required=True,
        help="regression runner script used in generated add_test commands",
    )
    parser.add_argument(
        "--python-executable",
        required=True,
        help="python executable used in generated add_test commands",
    )
    parser.add_argument(
        "--tool", required=True, help="tool executable path + optional args"
    )
    parser.add_argument(
        "--modes", nargs="+", required=True, help="supported test modes"
    )
    parser.add_argument("--timeout", type=int, required=False, help="per-test timeout")
    parser.add_argument(
        "--memory-limit",
        type=int,
        required=False,
        help="per-test virtual memory limit in megabytes",
    )
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=[],
        help="relative directory prefix to include below the root",
    )
    parser.add_argument(
        "--ignore-prefix",
        action="append",
        default=[],
        help="relative directory prefix to ignore below the root",
    )
    parser.add_argument(
        "--benchbringup",
        default=False,
        action="store_true",
        help="Propagate the benchmark bring-up flag to generated tests.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run discovery from the CLI, with optional compatibility subcommand."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "test-discover":
        argv = argv[1:]

    main_args = _arg_parser().parse_args(argv)
    output = generate_ctest_discovery(
        root_dir=main_args.root,
        runner=main_args.runner,
        python_executable=main_args.python_executable,
        tool=main_args.tool,
        modes=main_args.modes,
        timeout=main_args.timeout,
        memory_limit=main_args.memory_limit,
        include_prefixes=main_args.include_prefix,
        ignore_prefixes=main_args.ignore_prefix,
        benchbringup=main_args.benchbringup,
    )
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
