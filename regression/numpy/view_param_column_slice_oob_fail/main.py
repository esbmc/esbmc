import numpy as np

def second(col):
    x = col[1]
    return x

a = np.array([[1, 2], [3, 4]])
v = a[0:1, 0]
r = second(v)
