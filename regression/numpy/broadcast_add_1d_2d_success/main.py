import numpy as np

c = np.add([10, 20, 30], [[1, 2, 3], [4, 5, 6]])
assert c[0][0] == 11
assert c[0][1] == 22
assert c[1][0] == 14
assert c[1][2] == 36
