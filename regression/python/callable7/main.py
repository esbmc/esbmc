from typing import Callable


def f(x: int) -> int:
    return x + 1


def g(x: int) -> int:
    return x + 2


h: Callable[[int], int]

if 1:
    h: Callable = f
else:
    h: Callable = g

assert h(3) >= 4
