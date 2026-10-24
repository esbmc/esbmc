import numpy as np

a = np.array([[1, 2], [3, 4]])
it = np.nditer(a, flags=["multi_index"])
for x in it:
    i, j = it.multi_index
    assert x == a[j, i]
