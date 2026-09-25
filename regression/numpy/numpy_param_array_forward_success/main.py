import numpy as np

def inner(a):
    return a[1][0]

def outer(a):
    return inner(a)

a = np.array([[1, 2], [3, 4]])
val = outer(a)
assert val == 3
