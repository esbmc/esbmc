# Issue #9312: moda.ValueError is not the built-in ValueError.
import moda


def run() -> int:
    try:
        raise moda.ValueError("x")
    except ValueError:
        return 1
    return 0


assert run() == 1
