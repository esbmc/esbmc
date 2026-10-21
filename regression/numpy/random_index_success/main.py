import numpy as np

a = np.random.randint(0, 5, size=(2, 3))
i = np.random.randint(0, 2)
x = a[i][1]

assert x >= 0
assert x < 5
assert a[1][2] < 5
