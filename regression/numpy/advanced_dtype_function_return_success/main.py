import numpy as np


def make():
    return np.arange(1.5, 4, dtype=np.int16)


a = make()
assert len(a) == 3
assert a[0] == 1 and a[2] == 3


def bump(x):
    x[0] = x[0] + 100


def doubled(x):
    return x + x


b = np.array([100, -100], dtype=np.int8)
d = doubled(b)
assert d[0] == -56
bump(b)
assert b[0] == -56

def plus_one(x):
    return x + 1


def head(x):
    return x[0]


wrapping = np.array([127], dtype=np.int8)
assert plus_one(wrapping[0]) == -128
assert head(wrapping) + 1 == -128


def same(x):
    return x


passed = same(wrapping)
assert passed[0] == 127
