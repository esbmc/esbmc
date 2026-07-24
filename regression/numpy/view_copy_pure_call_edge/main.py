import numpy as np

x = np.array([[1, 2], [3, 4]])
row = x[0]
size = len(row)

assert size == 2
assert x[0][0] == 1
