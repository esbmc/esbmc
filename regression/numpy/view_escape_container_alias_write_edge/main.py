import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
box = [a[0]]
v = box[0]
v[1] = 9
assert a[0][1] == 9
box[0][2] = 8
assert a[0][2] == 8
