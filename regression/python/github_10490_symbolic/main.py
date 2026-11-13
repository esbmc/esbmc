# Issue #10490: a repeat count below one gives an empty list, however large.
from esbmc import nondet_int

def f() -> None:
    n = nondet_int()
    if n > 4:
        return
    a = [3] * n
    assert len(a) == (n if n > 0 else 0)

f()
