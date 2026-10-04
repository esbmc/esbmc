import numpy as np

c = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
v = c[1]
total = 0
for x in np.nditer(v):
    total += x
assert total == 26
