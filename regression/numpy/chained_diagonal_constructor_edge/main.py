import numpy as np

a = np.full((2, 2), 5).diagonal()

assert a[0] == 5
assert a[1] == 5
