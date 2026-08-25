# The same runtime-chosen callable, bound to a variable the annotation pass
# types for us. A bare `Callable` returns void, so a call through it used to be
# nondet -- worse than leaving the return unannotated (#6640).
from typing import Callable


def inc(m: int) -> int:
    return m + 1


def century(m: int) -> int:
    return m + 100


def pick(c: bool) -> Callable[[int], int]:
    return inc if c else century


h = pick(True)
assert h(1) == 2

k = pick(False)
assert k(1) == 101
