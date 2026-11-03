import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
for x, y in np.nditer([a, b], op_flags=[["readonly"], ["readwrite"]]):
    y[...] = x
    x[...] = 0
