import numpy as np

a = np.array([100, -100], dtype=np.int8)
first = a[0]
twice = first + first
assert twice == 200
