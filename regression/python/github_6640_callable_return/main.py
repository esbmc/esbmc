# A callable chosen at runtime, returned, and then called through the variable
# holding it. The spelled signature is what lets the call recover its return
# type; both branches of the choice must resolve to the right function (#6640).
from typing import Callable


def inc(m: int) -> int:
    return m + 1


def century(m: int) -> int:
    return m + 100


def pick(c: bool) -> Callable[[int], int]:
    return inc if c else century


h: Callable[[int], int] = pick(True)
assert h(1) == 2

k: Callable[[int], int] = pick(False)
assert k(1) == 101
