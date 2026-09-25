# pick(True) selects inc, so h(1) is 2. Resolving the function pointer to the
# other branch -- or leaving it unconstrained -- must not prove 101 (#6640).
from typing import Callable


def inc(m: int) -> int:
    return m + 1


def century(m: int) -> int:
    return m + 100


def pick(c: bool) -> Callable[[int], int]:
    return inc if c else century


h: Callable[[int], int] = pick(True)
assert h(1) == 101

k: Callable[[int], int] = pick(False)
assert k(1) == 101
