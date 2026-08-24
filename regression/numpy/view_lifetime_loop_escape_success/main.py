import numpy as np

n = nondet_bool()
a = np.array([[1, 2], [3, 4]])

while n:
    row = a[0]
    n = False

a[0][0] = 99
