import numpy as np

a = np.array([5])
b = np.broadcast_to(a, (2, 2))

acc = 0
for x in np.nditer(b):
    acc = acc + x

assert acc == 20
