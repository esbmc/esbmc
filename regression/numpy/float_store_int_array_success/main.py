import numpy as np

a = np.array([1, 2, 3])
a[0] = 2.7
a[1] = -1.9
i = np.random.randint(0, 3)
a[i] = 0.5
nan_rejected = False
try:
    a[2] = float("nan")
except ValueError:
    nan_rejected = True

assert a[i] == 0
assert nan_rejected
