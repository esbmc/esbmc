import numpy as np

a = np.zeros((2, 2), dtype=int).flatten()

assert a[0] == 0
assert a[1] == 0
assert a[2] == 0
assert a[3] == 0
