# Issue #9312: b.E is not the class a.E, though both are named E.
import a
import b
def run() -> int:
    try:
        raise a.E("x")
    except b.E:
        return 1
    return 0
assert run() == 1
