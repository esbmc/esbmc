import numpy as np

c = np.add([[1]], [[10, 20], [30, 40]])
assert c[0][0] == 11
assert c[0][1] == 21
assert c[1][0] == 31
assert c[1][1] == 41
