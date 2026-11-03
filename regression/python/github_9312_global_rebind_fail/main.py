# Issue #9312: a.rebind() makes a.E name ValueError.
import a
a.rebind()
def run() -> int:
    try:
        raise ValueError("v")
    except a.E:
        return 1
    except Exception:
        return 2
    return 0
assert run() == 2
