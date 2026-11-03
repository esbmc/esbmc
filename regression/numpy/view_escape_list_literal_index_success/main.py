import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
box = [a[0], a[1]]
assert box[0][1] == 2
assert box[1][2] == 6
assert box[-1][0] == 4
