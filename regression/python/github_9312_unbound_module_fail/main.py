# Issue #9312: `subprocess` is not bound, so the except clause raises NameError.
from subprocess import SubprocessError


def run() -> int:
    try:
        raise SubprocessError("x")
    except subprocess.SubprocessError:
        return 1
    return 0


assert run() == 1
