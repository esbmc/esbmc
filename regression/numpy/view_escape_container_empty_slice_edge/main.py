import numpy as np

a = np.array([10, 20, 30, 40, 50])
box = [a[2:2], a[1:3]]
assert len(box[0]) == 0
assert box[0].shape == (0,)
assert len(box[1]) == 2
