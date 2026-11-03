import numpy as np

def bump_row(row):
    old = row[1]
    row[1] = old + 5
    return row[1]

a = np.array([[1, 2], [3, 4]])
r = a[0]
result = bump_row(r)
assert result == 7
assert a[0][1] == 7
