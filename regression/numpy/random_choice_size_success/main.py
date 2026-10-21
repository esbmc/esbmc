import numpy as np

a = np.random.choice([1, 4, 9], size=4)

assert a.shape == (4,)
assert a[3] == 1 or a[3] == 4 or a[3] == 9
