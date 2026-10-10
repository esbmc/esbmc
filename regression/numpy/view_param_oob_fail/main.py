import numpy as np

def oob_access(row):
    return row[5]

a = np.array([1, 2, 3])
v = a[0:]
result = oob_access(v)
