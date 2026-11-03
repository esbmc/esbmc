import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
row = np.array([10, 20, 30])
total = 0
for x, y in np.nditer([a, row]):
    assert y == 10 * (1 + (x - 1) % 3)
    total += y
assert total == 120

column = np.array([[100], [200]])
pairs = 0
for x, y in np.nditer([a, column]):
    assert y == 100 if x <= 3 else y == 200
    pairs += 1
assert pairs == 6
