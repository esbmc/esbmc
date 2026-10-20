import numpy as np

def get_list(row):
    lst = row.tolist()
    return lst[0]

a = np.array([10, 20, 30])
v = a[0:]
result = get_list(v)
assert result == 10
