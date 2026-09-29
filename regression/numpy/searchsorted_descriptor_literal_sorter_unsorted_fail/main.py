import numpy as np

def make():
    return np.array([5, 1, 3])

idx = np.searchsorted(make(), 2, sorter=[0, 1, 2])
assert idx == 1
