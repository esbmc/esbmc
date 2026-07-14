# sorted() over a list of tuples whose components are symbolic (function
# parameters). The runtime sort model compares the tuple storage as
# reinterpreted integers — not Python's lexicographic order — and retypes the
# result elements as int, so element access on the result raised "int not
# subscriptable" and tie-breaking on a non-leading component was wrong. These
# lists are now sorted by a convert-time lexicographic sorting network that
# preserves the tuple element type.
from typing import List, Tuple


def two_elem(a: int, b: int) -> None:
    xs: List[Tuple[int, int]] = [(a, b), (b, a)]
    s = sorted(xs)
    # Result is ordered and element access works.
    assert s[0] <= s[1]
    assert s[0][0] <= s[1][0]


def tie_breaks_lexicographically(a: int, b: int) -> None:
    # Leading components tie: the second component decides the order.
    xs: List[Tuple[int, int]] = [(1, a), (1, b)]
    s = sorted(xs)
    assert s[0][1] <= s[1][1]


def three_elem_min(a: int, b: int, c: int) -> None:
    xs: List[Tuple[int, int]] = [(a, 9), (b, 9), (c, 9)]
    s = sorted(xs)
    lo = a
    if b < lo:
        lo = b
    if c < lo:
        lo = c
    assert s[0][0] == lo
    assert s[0][0] <= s[1][0] and s[1][0] <= s[2][0]


def reverse_order(a: int, b: int) -> None:
    xs: List[Tuple[int, int]] = [(1, a), (1, b)]
    s = sorted(xs, reverse=True)
    assert s[0][1] >= s[1][1]


two_elem(3, 1)
tie_breaks_lexicographically(5, 2)
three_elem_min(3, 1, 2)
reverse_order(2, 5)
