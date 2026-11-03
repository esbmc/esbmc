# Issue #9312: a.E does not catch b's E imported by name.
import a
from b import E
def run() -> int:
    try:
        raise E("b")
    except a.E:
        return 1
    return 0
assert run() == 1
