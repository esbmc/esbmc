from typing import Optional


def foo(x: Optional[int]) -> int:
    assert x is None or x is not None
    if x is None:
        return 0
    else:
        return x + 1


foo(None)
foo(42)
