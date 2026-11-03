import numpy as np

a = np.random.choice([1, 2, 3], size=2, replace=False)

assert a[0] < a[1]
