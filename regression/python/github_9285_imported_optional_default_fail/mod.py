from typing import Optional


def func(x: Optional[int] = None):
    if x is None:
        return "a"
    return x
