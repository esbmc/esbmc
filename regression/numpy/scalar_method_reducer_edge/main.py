import numpy as np

a = np.array([1.0, 2.0])

assert a[1].sum() == 2.0
assert a[0].argmax() == 0
assert a[1].std() == 0.0
