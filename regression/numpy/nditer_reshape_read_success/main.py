import numpy as np

a = np.array([1, 2, 3, 4])
r = np.reshape(a, (2, 2))

acc = 0
for x in np.nditer(r):
    acc = acc * 10 + x

assert acc == 1234
