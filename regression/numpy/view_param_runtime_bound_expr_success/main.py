import numpy as np

def zero_last(row):
    row[len(row) - 1] = 0
    return len(row)

a = np.array([1, 2, 3, 4])
n = 3
v = a[1:n]
x = zero_last(v) + 1
assert x == 3
assert a[2] == 0
w = a[0:n]
assert zero_last(w) == 3
y = 1 + zero_last(w)
assert y == 4
if zero_last(w) == 3:
    a[3] = 9
assert a[3] == 9
