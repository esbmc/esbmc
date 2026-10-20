import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
box = [a[0]]
same = box
assert same[0][1] == 2
assert same[0].shape == (3,)
