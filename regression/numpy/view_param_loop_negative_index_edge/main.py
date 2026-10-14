import numpy as np

def last_element(row):
    return row[-1]

a = np.array([10, 20, 30])
v = a[0:]
result = last_element(v)
assert result == 30
