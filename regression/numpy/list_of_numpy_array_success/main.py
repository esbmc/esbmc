import numpy as np

x = np.arange(3)
assert list(x) == [0, 1, 2]

y = np.linspace(0.0, 1.0, 3)
l = list(y)
assert len(l) == 3
assert l[1] == 0.5

z = np.array([4, 5, 6])
assert list(z) == [4, 5, 6]
