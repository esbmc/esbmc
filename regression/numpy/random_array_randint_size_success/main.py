import numpy as np

a = np.random.randint(3, 6, size=2)

assert len(a) == 2
assert a[0] >= 3
assert a[0] < 6
assert a[1] >= 3
assert a[1] < 6
