import numpy as np

calls = 0


def register(v):
    global calls
    calls = calls + 1


a = np.array([[1, 2, 3], [4, 5, 6]])
row = a[0]
register(row)
a[0][0] = 9
