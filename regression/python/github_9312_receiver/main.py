# Issue #9312: a local binding shadows the module, so the dotted type stays refused.
import subprocess

class Holder:
    SubprocessError = ValueError

def run() -> int:
    subprocess = Holder()
    try:
        raise ValueError("v")
    except subprocess.SubprocessError:
        return 1
    return 0

assert run() == 1
