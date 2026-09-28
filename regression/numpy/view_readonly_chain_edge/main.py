import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
row = a[0]
tail = row[1:]
n = len(tail)

assert n == 2
