import numpy as np

def first_elem(a):
    return a[0][0][0]

a = np.zeros((2, 2, 2))
assert first_elem(a) == 0
