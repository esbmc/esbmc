from typing import Callable

def square(x: int) -> int:
    return x * x

g: Callable[[int], int] = square

assert g(3) == 8
