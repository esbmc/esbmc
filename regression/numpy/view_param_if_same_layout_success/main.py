import numpy as np

def cond_write(row, flag):
    if flag:
        row[0] = 1
    else:
        row[0] = 2
    return row[0]

a = np.array([0, 0])
v = a[0:]
result = cond_write(v, True)
assert result == 1
