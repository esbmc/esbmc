#!/usr/bin/env python3

import argparse
from collections.abc import Sequence
from glob import iglob
import os
from pathlib import Path
import sys
from testing_model import TestDescription


def _is_excluded(
    path: Path,
    ignore_paths: list[Path],
) -> bool:
    for ignore_path in ignore_paths:
        if path.is_relative_to(ignore_path):
            return True
    return False


def _default_labels(relative_dir: Path) -> tuple[str, str]:
    assert not relative_dir.is_absolute()
    return ("regression", f"{relative_dir.parent.as_posix()}/")


def _cmake_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _cmake_join(arguments: Sequence[str]) -> str:
    return " ".join(_cmake_quote(argument) for argument in arguments)


def discover_tests(
    root_dir: str | Path,
    include_prefixes: Sequence[str],
    ignore_prefixes: Sequence[str],
) -> list[TestDescription]:
    """Recursively discover test.desc files below root_dir."""
    root_dir = Path(root_dir).absolute()
    # this should have been a set, but we want to preserve order for better readability of the generated CTest files...
    included_paths: list[Path] = [
        root_dir.joinpath(prefix).absolute()
        for prefix in (include_prefixes or [])
        if prefix
    ]
    ignored_paths: list[Path] = [
        root_dir.joinpath(prefix).absolute()
        for prefix in (ignore_prefixes or [])
        if prefix
    ]
    assert not set(included_paths).intersection(
        set(ignored_paths)
    ), f"Include and ignore prefixes must not overlap exactly: {included_paths} vs {ignored_paths}"

    discovered: list[TestDescription] = []

    for included_path in included_paths:
        discovered_one_dir: list[TestDescription] = []
        for test_desc_file in iglob(
            str(included_path / "**" / "test.desc"), recursive=True
        ):
            current_dir: Path = Path(test_desc_file).parent
            if _is_excluded(current_dir, ignored_paths):
                continue
            test = TestDescription.parse_test_description(current_dir, root_dir)
            discovered_one_dir.append(test)
        discovered_one_dir.sort(key=lambda test: test.relative_dir)
        discovered.extend(discovered_one_dir)
    return discovered


def generate_ctest_discovery(
    root_dir: str | Path,
    runner: str | Path,
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
    root_dir = Path(root_dir).resolve()
    runner = Path(runner).resolve()
    tests: list[TestDescription] = discover_tests(
        root_dir, include_prefixes, ignore_prefixes
    )

    lines: list[str] = []
    for test in tests:
        relative_dir = test.relative_dir
        relative_dir_text = relative_dir.as_posix()
        test_name = f"regression/{relative_dir_text}"

        command: list[str] = [
            python_executable,
            os.fspath(runner),
            f"--tool={tool}",
            f"--regression={test.test_dir.parent.absolute()}",
            "--modes",
            *modes,
            f"--file={test.test_dir.name}",
        ]
        if timeout is not None:
            command.append(f"--timeout={timeout}")
        if memory_limit is not None:
            command.append(f"--memory-limit={memory_limit}")
        if benchbringup:
            command.append("--benchbringup")

        lines.append(f"add_test({_cmake_join([test_name, *command])})")

        properties: list[str] = ["SKIP_RETURN_CODE", "10"]
        if timeout is not None:
            properties.extend(["TIMEOUT", str(timeout)])

        labels = ";".join(test.labels + _default_labels(relative_dir))
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
