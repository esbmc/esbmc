import numpy as np


def tail(v):
    return v[1::2]


def mutate(v):
    v[0] = 9
    return v[1]


a = np.array([1, 2, 3, 4])
assert mutate(tail(a)) == 4
assert a[1] == 2
