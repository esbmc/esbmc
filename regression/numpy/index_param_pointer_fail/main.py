# Indexing a numpy-array function parameter: the parameter decays to a pointer
# to its element type, so `a[i]` lowers to `*(a + i)` (pointer arithmetic) built
# natively in IREP2 (add2t over a pointer lhs) rather than a direct array index.
import numpy as np


def first(a) -> int:
    return a[0]


def third(a) -> int:
    return a[2]


d = np.array([7, 8, 9])
assert first(d) == 7
assert third(d) == 0  # must fail
