import numpy as np

def setk(row, k):
    row[0] = k
    return row[0]

a = np.array([10, 20, 30])
view = a[0:]
r = setk(k=5, row=view)
assert r == 5
assert a[0] == 10  # wrong: the callee wrote 5
