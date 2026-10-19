import numpy as np

z = np.zeros((2, 2), dtype="int8")
o = np.ones(3, dtype=np.uint16)
assert z[1][1] == 0
assert o.sum() == 3
