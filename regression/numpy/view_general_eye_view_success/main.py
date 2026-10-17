import numpy as np

def bump(row):
    row[0] = row[0] + 5
    return len(row)

e = np.eye(3)
row = e[1]
n = bump(row)
assert n == 3
assert e[1][0] == 5
assert e[1][1] == 1
col = e[:, 2]
assert col[2] == 1
assert max(row) == 5
