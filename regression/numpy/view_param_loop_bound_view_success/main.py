import numpy as np
def setall(row, x):
    i = 0
    while i < len(row):
        row[i] = x
        i = i + 1
a = np.array([1, 2, 3, 4])
m = 1
for n in range(2, 4):
    v = a[m:n]
    setall(v, 0)
assert a[0] == 1
assert a[1] == 0
assert a[2] == 0
assert a[3] == 4
