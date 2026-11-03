import numpy as np

a = np.array([1, 2, 3, 4])
x = a[1:3]
b = a
a = np.array([9, 9, 9, 9])
x[0] = 7
total = 0
for v in np.nditer(b):
    total = total * 10 + v
assert total == 1734
assert a[1] == 9
