import numpy as np


def first(a):
    return a[0]


x = np.zeros(3)
y = np.arange(3)
assert first(x) == 0.0
assert first(y) == 0
