import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.transpose(a)
a[1][0] = 8

acc = 0
for x in np.nditer(t):
    acc = acc * 10 + x

assert acc == 1824
