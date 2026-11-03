import numpy as np


def one():
    return 1


# One call, used once: inlining it changes nothing.
def pick(v):
    unused = 5
    n = one()
    return v[n]


a = np.array([[1, 2, 3], [4, 5, 6]])
row = a[0]
assert pick(row) == 2
a[0][1] = 9
assert pick(row) == 9
