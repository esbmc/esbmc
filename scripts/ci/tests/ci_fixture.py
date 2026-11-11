"""Real CMake projects for the CI helper tests."""

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmake_list(items):
    return ";".join(items)


def write_project(root, tests, extra=()):
    """A CMake project registering `tests`: dicts of name, command, labels, props."""
    lines = [
        "cmake_minimum_required(VERSION 3.19)", "project(ci_fixture NONE)", "enable_testing()",
        'set(ESBMC_REGRESS_TIMEOUT 120 CACHE STRING "")', *extra
    ]
    for test in tests:
        command = " ".join(f'"{part}"' for part in test["command"])
        lines.append(f'add_test(NAME "{test["name"]}" COMMAND {command})')
        props = dict(test.get("props", {}))
        if test.get("labels"):
            props["LABELS"] = cmake_list(test["labels"])
        for key, value in props.items():
            lines.append(f'set_tests_properties("{test["name"]}" PROPERTIES {key} "{value}")')
    with open(os.path.join(root, "CMakeLists.txt"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def configure(root):
    build = os.path.join(root, "build")
    subprocess.run(["cmake", "-S", root, "-B", build], check=True, capture_output=True)
    return build


def project(tests, extra=()):
    """A configured project in a fresh directory: (tempdir handle, build dir)."""
    tmp = tempfile.TemporaryDirectory()
    write_project(tmp.name, tests, extra)
    return tmp, configure(tmp.name)


def write_case(suite, name, mode, requires=()):
    """A regression test directory the harness's TestCase can parse."""
    path = os.path.join(suite, name)
    os.makedirs(path)
    with open(os.path.join(path, "main.c"), "w", encoding="utf-8") as fp:
        fp.write("int main() { return 0; }\n")
    body = [mode, "main.c", "--incremental-bmc"]
    body += [f"REQUIRES {capability}" for capability in requires]
    body.append("^VERIFICATION SUCCESSFUL$")
    with open(os.path.join(path, "test.desc"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(body) + "\n")


def harness_command(exit_code, suite, name, capabilities="", tool="esbmc"):
    """A command shaped like add_esbmc_regression_test's, exiting with `exit_code`."""
    return [
        "/bin/sh", "-c", f"exit {exit_code}", "--", f"--tool={tool}", f"--regression={suite}",
        "--modes", "CORE", f"--file={name}", f"--capabilities={capabilities}"
    ]


def regression(name, suite_label, command=("/bin/true", ), extra_labels=()):
    return {
        "name": name,
        "command": list(command),
        "labels": ["regression", suite_label, *extra_labels],
        "props": {
            "SKIP_RETURN_CODE": "10",
            "ENVIRONMENT": "ESBMC_REGRESS_TIMEOUT=120"
        },
    }
