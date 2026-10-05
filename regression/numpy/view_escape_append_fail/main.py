import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
row = a[0]
items = []
items.append(row)
items[0][1] = 9
assert a[0][1] == 9
