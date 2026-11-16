import numpy as np


def first(a):
    return a[0]


h = lambda: first(np.zeros(2))
assert h() == 0.0

c = nondet_bool()
assert (c and first(np.ones(2)) == 1.0) or (not c and first(np.zeros(2)) == 0.0)
