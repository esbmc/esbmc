# Issue #9312: an except clause naming module.Class.
import subprocess

def run() -> int:
    try:
        raise subprocess.SubprocessError("base")
    except subprocess.CalledProcessError:
        return 1
    return 0

assert run() == 1
