import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[10, 20], [30, 40]])
total = 0
for x, y in np.nditer([a, b]):
    assert y == 10 * x
    total += x + y
assert total == 110

it = np.nditer([a, b])
assert it[0] == 1 and it[1] == 10
