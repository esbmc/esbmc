from typing import Any


def f(b: bool) -> bool:
    return isinstance(b, int)


x = True
y: Any = False
n = 3
assert isinstance(x, int)
assert isinstance(y, int)
assert f(False)
assert not isinstance(x, float)
assert not isinstance(n, bool)
