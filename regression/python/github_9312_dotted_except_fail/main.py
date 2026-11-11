# Issue #9312: a subclass handler does not catch its base class.
import subprocess


def run(x: int) -> int:
    try:
        if x > 0:
            raise subprocess.SubprocessError("boom")
        return 0
    except subprocess.CalledProcessError:
        return 1


assert run(1) == 1
