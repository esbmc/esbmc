# Issue #9312: a.E does not catch the E local to b.go().
import a
import b
def run() -> int:
    try:
        b.go()
    except a.E:
        return 1
    return 0
assert run() == 1
