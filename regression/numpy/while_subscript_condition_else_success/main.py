import numpy as np
a = np.array([1, 2, 0, 4])
i = 0
hit = 0
while a[i] != 0:
    i += 1
    if i == 9:
        break
else:
    hit = 1
assert i == 2
assert hit == 1
