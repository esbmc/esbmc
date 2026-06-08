import numpy as np

c = np.add([[1, 2, 3]], [[10], [20]])
assert c[0][0] == 11
assert c[0][2] == 13
assert c[1][0] == 21
assert c[1][2] == 23
