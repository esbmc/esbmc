import numpy as np

a = np.random.choice((2.5, 3.5), size=(2, 3))

assert a.shape == (2, 3)
assert a[1][2] == 2.5 or a[1][2] == 3.5
