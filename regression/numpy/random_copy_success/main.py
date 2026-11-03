import numpy as np

a = np.random.random(3)
b = a.copy()
c = np.copy(a)
items = a.tolist()

assert b[0] == a[0]
assert c[2] < 1.0
assert items[1] == a[1]
