import numpy as np

def total(v):
    return v[0] + v[1] + len(v)

def tail_total(row):
    sub = row[2:]
    return total(sub)

a = np.array([1, 2, 3, 4])
v = a[0:]
assert tail_total(v) == 9
