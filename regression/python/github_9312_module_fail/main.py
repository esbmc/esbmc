# Issue #9312: an except clause naming module.Class.
import helper

def run(x: int) -> int:
    try:
        return helper.go(x)
    except helper.HelperError:
        return -1

assert run(3) == 3
