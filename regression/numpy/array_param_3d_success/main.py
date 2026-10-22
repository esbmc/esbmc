import numpy as np

def first_elem(a):
    return a[0][0][0]

a = np.zeros((2, 2, 2))
assert first_elem(a) == 0
a[1][1][1] = 5
assert a[1][1][1] == 5
