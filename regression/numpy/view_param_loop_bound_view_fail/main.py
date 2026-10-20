import numpy as np
def setall(row, x):
    i = 0
    while i < len(row):
        row[i] = x
        i = i + 1
a = np.array([1, 2, 3, 4])
for n in range(2, 4):
    v = a[1:n]
    setall(v, 0)
assert a[3] == 0
