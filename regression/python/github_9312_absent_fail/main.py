# Issue #9312: subprocess has no attribute ValueError.
import subprocess
def run() -> int:
    try:
        raise ValueError("v")
    except subprocess.ValueError:
        return 1
    return 0
assert run() == 1
