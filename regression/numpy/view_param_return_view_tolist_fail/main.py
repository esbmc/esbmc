import numpy as np

def bump(row):
    row[1] = row[1] + 10
    return row

a = np.array([1, 2, 3, 4])
v = a[1:3]
l = bump(v).tolist()
assert l[1] == 3
