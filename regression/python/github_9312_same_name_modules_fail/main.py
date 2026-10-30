# Issue #9312: modb.Error is not moda.Error, though both share a class symbol.
import moda
import modb


def run() -> int:
    try:
        raise moda.Error("a")
    except modb.Error:
        return 1
    return 0


assert run() == 1
