# Negative variant: after sorting a list of tuples with symbolic components,
# the assertion claims the wrong (descending) order on the tie-broken second
# component. The convert-time lexicographic sort refutes it — confirming the
# sort is genuinely lexicographic and not a reinterpreted-integer comparison.
from typing import List, Tuple


def tie_wrong(a: int, b: int) -> None:
    xs: List[Tuple[int, int]] = [(1, a), (1, b)]
    s = sorted(xs)
    assert s[0][1] > s[1][1]


tie_wrong(5, 2)
