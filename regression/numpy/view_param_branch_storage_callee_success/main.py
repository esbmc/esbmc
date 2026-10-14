import numpy as np

def read_first(row):
    n = len(row)
    return row[0]

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
flag = True
if flag:
    v = a[0:]
else:
    v = b[0:]
result = read_first(v)
assert result == 1
