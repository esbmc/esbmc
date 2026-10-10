import numpy as np

def increment(row):
    row[0] = row[0] + 1
    return row[0]

a = np.array([10, 20, 30])
result = increment(a[0:])
assert result == 10
