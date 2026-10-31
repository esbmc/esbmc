import numpy as np

base = np.array([[1, 2], [3, 4]], dtype=int)
lst = base.tolist()

assert lst[0][0] == 1
assert lst[0][1] == 2
assert lst[1][0] == 3
assert lst[1][1] == 4

narrow = np.array([100, -100], dtype=np.int8)
narrow[0] += 100
assert narrow.tolist()[0] == -56
single = np.array([0.1], dtype=np.float32)
assert single.tolist()[0] != 0.1
