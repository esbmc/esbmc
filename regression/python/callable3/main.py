from typing import Callable

def f(x: int) -> int:
    return x + 1

def g(x: int) -> int:
    return x * 2

h: Callable[[int], int] = f

assert g(3) == 6

