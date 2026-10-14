import numpy as np

a = np.array([[1, 2], [3, 4]])
row = a[0]
shared = np.shares_memory(a, row)

assert shared
