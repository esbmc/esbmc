from typing import Callable


def inc(x: int) -> int:
    return x + 1


g: Callable[[int], int] = inc

assert g(1) == 2
assert g(5) == 6
