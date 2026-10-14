import numpy as np

c = np.add([[1, 2, 3], [4, 5, 6]], [10, 20, 30])
assert c[0][0] == 11
assert c[0][2] == 33
assert c[1][1] == 25
assert c[1][2] == 36
