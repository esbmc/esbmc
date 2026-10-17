import numpy as np

a = np.array([1, 2, 3])
v = a[0:]
count = lambda row: len(row)
x = count(v)
assert x == 3
