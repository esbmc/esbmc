# Issue #9312: subprocess names os.path, not the subprocess module.
from os import path as subprocess
import subprocess as sp

def run() -> int:
    try:
        raise sp.SubprocessError("x")
    except subprocess.SubprocessError:
        return 1

assert run() == 1
