import numpy as np


def total(a):
    s = 0.0
    for v in a:
        s += v
    return s


def rows(a):
    return a.shape[0] * 10 + a.shape[1]


def set_first(a):
    a[0] = 9


x = np.arange(1.0, 4.0)
assert total(x) == 6.0
y = np.full((2, 3), 3)
assert rows(y) == 23
w = np.zeros(3, dtype=int)
set_first(w)
assert w[0] == 9
