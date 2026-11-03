import numpy as np

def last(row):
    k = len(row)
    return row[k]

a = np.array([1, 2, 3, 4])
n = 3
v = a[0:n]
x = last(v)
