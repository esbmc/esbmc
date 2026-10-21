import numpy as np

a = np.array([1, 2])
a[0] = 2.7

assert a[0] == 3
