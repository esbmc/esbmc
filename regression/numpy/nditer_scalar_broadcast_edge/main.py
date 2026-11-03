import numpy as np

a = np.array([[1, 2], [3, 4]])
total = 0
for x, s in np.nditer([a, 5]):
    assert s == 5
    total += x * s
assert total == 50
