import numpy as np

a = np.array([1, 2, 3])
empty = a[1:1]
items = empty.tolist()

assert len(items) == 0

