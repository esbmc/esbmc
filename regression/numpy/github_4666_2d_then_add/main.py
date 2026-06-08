import numpy as np

a = np.array([[1, 2], [3, 4]])
x = a[1, 0]
y = x + 7

assert y == 10
