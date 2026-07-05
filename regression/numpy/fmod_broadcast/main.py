import numpy as np

a = np.fmod([[5.5, 7.0], [8.0, 9.0]], [2.0, 3.0])

assert a[0][0] == 1.5
assert a[0][1] == 1.0
assert a[1][0] == 0.0
assert a[1][1] == 0.0
