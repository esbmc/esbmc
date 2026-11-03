# Issue #9312: an except clause naming a module's exception class.
import subprocess


def run(x: int) -> int:
    try:
        if x > 0:
            raise subprocess.SubprocessError("boom")
        return 0
    except subprocess.SubprocessError:
        return 1


def run_tuple(x: int) -> int:
    try:
        if x > 0:
            raise subprocess.CalledProcessError(1, "cmd")
        return 0
    except (OSError, subprocess.SubprocessError):
        return 2


assert run(1) == 1
assert run(0) == 0
assert run_tuple(1) == 2
