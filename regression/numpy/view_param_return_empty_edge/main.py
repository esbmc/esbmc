import numpy as np

def get_len(row):
    return len(row)

a = np.array([1, 2, 3])
v = a[0:0]
result = get_len(v)
assert result == 0
