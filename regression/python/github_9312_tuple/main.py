# Issue #9312: an except clause naming module.Class.
import subprocess

def run(x: int) -> int:
    try:
        if x > 0:
            raise subprocess.CalledProcessError(2, "ls")
        raise OSError("io")
    except (OSError, subprocess.SubprocessError):
        return 1

assert run(1) == 1
assert run(0) == 1
