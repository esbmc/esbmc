import numpy as np

a = np.array([1000.0])
b = np.exp(a)
assert b[0] < 1e308, "overflow"
