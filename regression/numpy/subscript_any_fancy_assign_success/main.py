import numpy as np

a = np.array([10, 20, 30, 40])
sel = a[[0, 2]]
assert sel[0] == 10
assert sel[1] == 30
