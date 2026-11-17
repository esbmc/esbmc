# Issue #10490: a repeat count below one gives an empty list, however large.
from esbmc import nondet_int

def f(l: list[int]) -> None:
    n = nondet_int()
    if n > 4:
        return
    a = l * n
    assert len(a) == 0

f([1, 2])
