# Issue #9312: a rebinds E to ValueError after creating an instance.
import a
def run() -> int:
    try:
        a.go()
    except a.E:
        return 1
    return 0
assert run() == 1
