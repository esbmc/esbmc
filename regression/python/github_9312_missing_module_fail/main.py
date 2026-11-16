# Issue #9312: exceptions is not an importable module.
import exceptions
def run() -> int:
    try:
        raise ValueError("v")
    except exceptions.ValueError:
        return 1
    return 0
assert run() == 1
