import numpy as np

f = np.array([1.5, 2.0])
caught = False
try:
    b = f & 1
except TypeError:
    caught = True
assert caught
assert (np.array([6, 3]) & 3)[0] == 2
