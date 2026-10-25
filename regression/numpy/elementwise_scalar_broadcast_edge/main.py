import numpy as np

m = np.array([[1, 2, 3], [4, 5, 6]])
d = 2.5 - m
assert d[0, 0] == 1.5 and d[1, 2] == -3.5
h = m // 2
assert h[0, 0] == 0 and h[1, 2] == 3
caught = False
try:
    m ** -2
except ValueError:
    caught = True
assert caught
