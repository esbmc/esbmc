import numpy as np

a = np.array([[1.9, 2.9], [3.9, 4.9]], dtype=np.int32)
t = a.transpose()
assert t[0][1] == 3
a[0][1] = 7
assert t[1][0] == 7
