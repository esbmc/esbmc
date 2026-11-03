import numpy as np

a = np.array([[1.7, -1.7]], dtype="int16")
t = a.T
assert a[0][1] == -1
assert t[1][0] == -1
