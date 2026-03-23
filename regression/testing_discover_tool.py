#!/usr/bin/env python3

import argparse
from collections.abc import Sequence
import os
from pathlib import Path, PurePosixPath
import sys
from testing_model import TestDescription

RelativePath = PurePosixPath


def _relative_path(path: str | Path) -> RelativePath:
    normalized = os.fspath(path).replace("\\", "/").strip("/")
    if normalized in {"", "."}:
        return RelativePath()
    return RelativePath(normalized)


def _relative_path_text(path: RelativePath | Path) -> str:
    relative_path: RelativePath = (
        _relative_path(path) if isinstance(path, Path) else path
    )
    return "" if relative_path == RelativePath() else relative_path.as_posix()


def _relative_to_root(root_dir: Path, path: Path) -> RelativePath:
    return _relative_path(path.relative_to(root_dir))


def _is_same_or_child(path: RelativePath, prefix: RelativePath) -> bool:
    if prefix == RelativePath():
        return True
    return path == prefix or prefix in path.parents


def _should_include_path(
    path: RelativePath,
    include_prefixes: Sequence[RelativePath],
    ignore_prefixes: Sequence[RelativePath],
) -> bool:
    if any(_is_same_or_child(path, prefix) for prefix in ignore_prefixes):
        return False
    if not include_prefixes:
        return True
    return any(_is_same_or_child(path, prefix) for prefix in include_prefixes)


def _should_descend_into(
    path: RelativePath,
    include_prefixes: Sequence[RelativePath],
    ignore_prefixes: Sequence[RelativePath],
) -> bool:
    if any(_is_same_or_child(path, prefix) for prefix in ignore_prefixes):
        return False
    if not include_prefixes:
        return True
    return any(
        _is_same_or_child(path, prefix) or _is_same_or_child(prefix, path)
        for prefix in include_prefixes
    )


def _suite_order_for_path(
    path: RelativePath, include_prefixes: Sequence[RelativePath]
) -> int:
    for index, prefix in enumerate(include_prefixes):
        if _is_same_or_child(path, prefix):
            return index
    return len(include_prefixes)


def _default_labels(relative_dir: RelativePath) -> tuple[str, str]:
    assert (
        relative_dir.parent != RelativePath()
    ), f"test directly under root has no suite directory: {relative_dir}"
    return ("regression", f"{_relative_path_text(relative_dir.parent)}/")


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
    root_dir = Path(root_dir).resolve()
    normalized_include_prefixes: list[RelativePath] = [
        _relative_path(prefix) for prefix in (include_prefixes or []) if prefix
    ]
    normalized_ignore_prefixes: list[RelativePath] = [
        _relative_path(prefix) for prefix in (ignore_prefixes or []) if prefix
    ]

    discovered: list[TestDescription] = []

    for current_dir_str, dirnames, filenames in os.walk(root_dir, topdown=True):
        current_dir = Path(current_dir_str)
        dirnames.sort()
        relative_dir = _relative_to_root(root_dir, current_dir)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if _should_descend_into(
                relative_dir / dirname,
                normalized_include_prefixes,
                normalized_ignore_prefixes,
            )
        ]

        if "test.desc" not in filenames:
            continue
        if not _should_include_path(
            relative_dir, normalized_include_prefixes, normalized_ignore_prefixes
        ):
            continue

        test = TestDescription.parse_test_description(current_dir, root_dir)
        discovered.append(test)

    discovered.sort(
        key=lambda test: (
            _suite_order_for_path(
                _relative_path(test.relative_dir), normalized_include_prefixes
            ),
            _relative_path_text(test.relative_dir),
        )
    )
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
        relative_dir = _relative_path(test.relative_dir)
        relative_dir_text = _relative_path_text(relative_dir)
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
