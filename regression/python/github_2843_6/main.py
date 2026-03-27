from typing import Any


def foo(x: int) -> Any:
    if x > 0:
        return x + 1
    else:
        return x - 1


assert foo(2) == 3
assert foo(-2) == -3
