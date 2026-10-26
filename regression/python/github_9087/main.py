# GitHub #9087: a Sequence[T] parameter holds the list passed to it.
from typing import Sequence


def second(xs: Sequence[str]) -> str:
    return xs[1]


def total(xs: Sequence[int]) -> int:
    t = 0
    for x in xs:
        t += x
    return t + len(xs)


assert second(["a", "b"]) == "b"
assert total([1, 2, 3]) == 9
