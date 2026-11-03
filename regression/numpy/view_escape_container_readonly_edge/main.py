import numpy as np

a = np.array([1, 2, 3])
box = [np.broadcast_to(a, (2, 3))]
assert box[0][1][2] == 3
box[0][0][0] = 9
