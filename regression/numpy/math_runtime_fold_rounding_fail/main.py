import numpy as np

a = np.array([1.0, 2.0])
b = np.exp(a)
assert b[0] == 2.718281828459045, "rounding"
