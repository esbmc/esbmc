import numpy as np

c = np.add([[1, 2, 3], [4, 5, 6]], [[10]])
assert c[0][0] == 11
assert c[0][2] == 13
assert c[1][0] == 14
assert c[1][2] == 16
