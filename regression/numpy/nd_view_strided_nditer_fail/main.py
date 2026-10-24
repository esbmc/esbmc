import numpy as np

a = np.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
b = a[:, :, 1]
total = 0
for x in np.nditer(b):
    total += x
assert total == 65
