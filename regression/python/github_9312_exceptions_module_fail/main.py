# Issue #9312: a module named exceptions does not hold the built-ins.
import exceptions


def run() -> int:
    try:
        raise ValueError("v")
    except exceptions.ValueError:
        return 1
    return 0


assert run() == 1
