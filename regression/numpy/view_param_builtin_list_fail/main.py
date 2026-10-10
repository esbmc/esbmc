import numpy as np

def to_list(row):
    lst = list(row)
    return lst[0]

a = np.array([10, 20, 30])
v = a[0:]
result = to_list(v)
assert result == 10
