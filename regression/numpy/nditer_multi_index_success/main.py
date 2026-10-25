import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
it = np.nditer(a, flags=["multi_index"])
total = 0
for x in it:
    i, j = it.multi_index
    assert x == a[i, j]
    total += x * (i + 1) * (j + 1)
assert total == 78

t = np.transpose(a)
it2 = np.nditer(t, flags=["multi_index"])
order = 0
for x in it2:
    i, j = it2.multi_index
    assert x == t[i, j]
    order = order * 10 + x
assert order == 123456
