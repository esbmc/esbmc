import numpy as np

a = np.array([10, 20, 30])
first = a[0:1]
assert first[0] == 10
last = a[2:3]
assert last[0] == 30
