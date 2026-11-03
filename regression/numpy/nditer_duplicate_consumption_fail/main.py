import numpy as np

a = np.array([1, 2, 3])
it = np.nditer(a)
for x in it:
    pass
count = 0
for x in it:
    count += 1
assert count == 3
