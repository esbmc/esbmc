# Negative variant: reading a list[Tuple[...]] element through a function
# return now yields the real tuple components, so a wrong claim about a
# component is refuted (rather than crashing or silently passing).
from typing import List, Tuple


def first_pair(xs: List[Tuple[int, int]]) -> Tuple[int, int]:
    return xs[0]


def main() -> None:
    h: List[Tuple[int, int]] = [(1, 2)]
    a, b = first_pair(h)
    assert a == 2


main()
