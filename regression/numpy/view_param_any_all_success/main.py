import numpy as np

def check_any_all(row):
    return any(row) and not all(row)

a = np.array([0, 1, 0])
v = a[0:]
result = check_any_all(v)
assert result == True
