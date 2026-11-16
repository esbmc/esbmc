# Issue #9312: other.py, verified alongside, binds E as well.
import m


def run() -> int:
    try:
        raise ValueError("v")
    except m.E:
        return 1
    return 0


assert run() == 1
