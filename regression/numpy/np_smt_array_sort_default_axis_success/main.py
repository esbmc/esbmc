import numpy as np

a = np.array([[3, 1], [4, 2]])
b = np.sort(a)  # Default axis=-1 sorts each row

assert b[0, 0] == 1
assert b[0, 1] == 3
assert b[1, 0] == 2
assert b[1, 1] == 4
