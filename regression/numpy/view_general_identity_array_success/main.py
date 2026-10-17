import numpy as np

i = np.identity(3)
assert i.shape == (3, 3)
assert i[2][2] == 1
assert i[0][2] == 0
