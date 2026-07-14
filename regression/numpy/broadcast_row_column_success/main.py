import numpy as np

c = np.add([1, 2, 3], [[10], [20], [30]])
assert c[0][0] == 11
assert c[1][1] == 22
assert c[2][2] == 33
