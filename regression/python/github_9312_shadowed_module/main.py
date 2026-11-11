# Issue #9312: a local named like the module is not the module.
import subprocess


class Fake:
    SubprocessError = ValueError


def run() -> int:
    subprocess = Fake()
    try:
        raise ValueError("x")
    except subprocess.SubprocessError:
        return 1
    return 0


assert run() == 1
