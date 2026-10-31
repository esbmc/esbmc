# Issue #9312: helper.ValueError is not the builtin ValueError.
import helper
def run() -> int:
    try:
        raise ValueError("builtin")
    except helper.ValueError:
        return 1
    return 0
assert run() == 1
