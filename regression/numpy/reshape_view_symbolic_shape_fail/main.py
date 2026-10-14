import numpy as np

n = int(input())
a = np.array([1, 2, 3, 4])
r = np.reshape(a, (n, 2))

assert r[0][0] == 1
