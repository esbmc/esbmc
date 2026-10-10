import numpy as np

def read_first(row):
    x = row[0]
    y = row[1]
    return x + y

a = np.array([10, 20, 30])
v = a[1:]
result = read_first(v)
assert result == 50
