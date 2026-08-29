import numpy as np

a = np.array([[1, 2], [3, 4]])
b = a.copy()
base = b.base

assert base is None
