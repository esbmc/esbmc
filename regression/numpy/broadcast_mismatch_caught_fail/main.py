import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5])
caught = False
try:
    c = a + b
except ValueError:
    caught = True
assert not caught
