import numpy as np

def sink(x):
    return len(x)

a = np.array([[1, 2], [3, 4]])
v = np.moveaxis(a, 0, 1)
assert sink(v) == 2
