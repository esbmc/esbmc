import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
box = (a[:, 1],)
assert box[0][0] == 2
assert box[0][1] == 5
assert len(box[0]) == 2
