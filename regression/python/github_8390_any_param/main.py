from typing import Any


def m(v, hint) -> bool:
    if hint is Any:
        return True
    return isinstance(v, hint)


assert m(1, Any)
assert m(1, int)
assert not m("a", int)
try:
    isinstance(1, Any)
    assert False
except TypeError:
    pass
