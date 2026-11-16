# Issue #9312: subprocess is imported only after run() is called.
def run() -> int:
    try:
        raise subprocess.SubprocessError("x")
    except subprocess.SubprocessError:
        return 1

assert run() == 1
import subprocess
