import numpy as np

# A single element read out of an array is a scalar, not a view: it can go
# into an ordinary container.
a = np.array([[1, 2, 3], [4, 5, 6]])
row = a[1]
items = [0, 0]
items[0] = a[0][2]
items.append(row[1])
pair = (a[1][0], row[2])
assert items[0] == 3
assert items[2] == 6
assert pair[0] + pair[1] == 10
