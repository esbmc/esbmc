# Issue #9312: an except clause naming module.Class.
import subprocess

def run() -> int:
    try:
        raise subprocess.CalledProcessError(2, "ls")
    except (OSError, ValueError):
        return 1

run()
