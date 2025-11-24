from typing import Any

def foo(x: float) -> Any:
    if x > 0:
        return x + 1
    else:
        return x - 1

assert foo(2.1) == 3.1
assert foo(-2.1) == -3.1
