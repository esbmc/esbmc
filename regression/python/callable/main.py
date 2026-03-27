from typing import Callable


def double(x: int) -> int:
    return x


g: Callable[[int], int] = double

assert g(1) == 1
