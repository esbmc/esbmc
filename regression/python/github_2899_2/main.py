from typing import Optional


def foo(x: Optional[int] = None) -> None:
    if x is not None:
        y = x


foo()
