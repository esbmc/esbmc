import numpy as np

def last_of_tail(row):
    sub = row[1:]
    sub[-1] = 9
    return sub[-2]

a = np.array([1, 2, 3, 4])
v = a[0:]
assert last_of_tail(v) == 3
assert a[3] == 9
