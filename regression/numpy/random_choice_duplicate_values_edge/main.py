import numpy as np

a = np.random.choice([5, 5, 7], size=2, replace=False)

assert a[0] == 5 or a[0] == 7
assert a[0] + a[1] >= 10
