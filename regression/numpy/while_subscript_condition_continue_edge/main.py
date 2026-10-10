import numpy as np
a = np.array([1, 2, 3, 0])
i = 0
s = 0
while a[i] != 0:
    i += 1
    if a[i - 1] == 2:
        continue
    s += a[i - 1]
assert s == 4
assert i == 3
