import numpy as np

a = np.array([1, 2, 3])
it = np.nditer(a)
x = next(it)
a = np.array([7, 8, 9])
y = next(it)
assert y == 8
