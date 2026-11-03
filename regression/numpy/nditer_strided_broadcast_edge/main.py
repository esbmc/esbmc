import numpy as np

a = np.arange(6)
s = a[::2]
b = np.array([[1], [2]])
it = np.nditer([s, b], flags=["multi_index", "c_index"])
assert it.shape == (2, 3)
total = 0
for x, y in it:
    total += int(x) * int(y)
    assert it.index == it.multi_index[0] * 3 + it.multi_index[1]
assert total == (0 + 2 + 4) * 3
