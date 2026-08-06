import numpy as np

v = np.expand_dims([1, 2], 0)

assert v[0][0] == 1
assert v[0][1] == 2
