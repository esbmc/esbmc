import numpy as np

a = np.random.randint(0, 3, size=3)

assert a[0] == a[1] or a[1] == a[2] or a[0] == a[2]
