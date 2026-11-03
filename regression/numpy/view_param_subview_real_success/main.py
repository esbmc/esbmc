import numpy as np

def tail_sees_write(row):
    sub = row[1:]
    row[1] = 50
    return sub[0]

a = np.array([10, 20, 30])
v = a[0:]
assert tail_sees_write(v) == 50
assert a[1] == 50
