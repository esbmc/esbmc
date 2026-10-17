# `pick()(x)` evaluates `pick()` where the call sits: not at all when `and`
# short-circuits, and once per iteration in a loop condition (#7802).
from typing import Callable

count: int = 0


def inc(m: int) -> int:
    return m + 1


def pick() -> Callable:
    global count
    count = count + 1
    return inc


def main() -> None:
    flag: bool = False
    r: bool = flag and pick()(1) == 2
    assert count == 0
    i: int = 0
    while pick()(i) < 3:
        i = i + 1
    assert count == 3


main()
