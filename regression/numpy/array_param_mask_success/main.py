import numpy as np

def select(a, mask):
    return a[mask]

a = np.array([1, 2, 3])
mask = np.array([True, False, True])
b = select(a, mask)
