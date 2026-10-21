import numpy as np

a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
v = a[1:]
b = np.array([[1, 2], [3, 4]])
i = np.random.randint(0, 2)

assert v[1].sum() == 11.0
assert v[1].max() == 6.0
assert a[2].sum() == 11.0
assert b[i].max() >= 2
