import numpy as np

a = np.array([127], dtype="i1")
a[0] += 1
assert a[0] == 128
