import numpy as np
a = np.array([1, 2, 3, 4, 5])
v = a[0:2]
i = 0
while i < 2:
    x = v[1]
    if i == 0:
        v = a[1:5]
    i = i + 1
