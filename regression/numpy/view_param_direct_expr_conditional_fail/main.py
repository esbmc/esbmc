import numpy as np
def setk(row, k):
    row[0] = k
    return row[0]
a = np.array([1, 2, 3, 4])
c = True
x = 1 if c else setk(a[0:2], 7)
assert a[0] == 7
