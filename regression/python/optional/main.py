from typing import Optional

def foo(x: int, y: Optional[int] = None) -> int:
    if y is None:
        y = 0
    return x + y

assert foo(1) == 1
