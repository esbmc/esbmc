import numpy as np

def passthrough(row):
    row[0] = 42
    return row

a = np.array([1, 2, 3, 4])
v = a[::2]
ret = passthrough(v)
assert a[0] == 42
assert ret[0] == 42
assert len(ret) == 2
ret[1] = 7
assert a[2] == 3  # wrong: ret aliases a[::2]
assert a[1] == 2
