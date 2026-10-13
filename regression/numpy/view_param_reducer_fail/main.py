import numpy as np

def sum_view(row):
    return np.sum(row)

a = np.array([1, 2, 3])
v = a[0:]
result = sum_view(v)
assert result == 0  # wrong: sum is 6
