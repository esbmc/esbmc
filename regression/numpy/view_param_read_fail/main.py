import numpy as np

def read_first(row):
    x = row[0]
    return x

a = np.array([10, 20, 30])
v = a[1:]
result = read_first(v)
assert result == 99
