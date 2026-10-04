import numpy as np


def tail_rows(m):
    i = nondet_int()
    __ESBMC_assume(i >= 0 and i <= 2)
    s = m[i:]
    assert len(s) == 3 - i
    if len(s) > 0:
        s[0][1] = 99
    return i


a = np.array([[1, 2], [3, 4], [5, 6]])
k = tail_rows(a)
assert a[k][1] == 99
