from typing import Callable

def apply(f: Callable[[int], int], x: int) -> int:
    return f(x)

def square(n: int) -> int:
    return n * n

assert apply(square, 4) == 16
