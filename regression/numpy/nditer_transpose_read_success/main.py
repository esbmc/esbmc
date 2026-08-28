import numpy as np

a = np.array([[1, 2], [3, 4]])
t = np.transpose(a)

acc = 0
for x in np.nditer(t):
    acc = acc * 10 + x

assert acc == 1324
