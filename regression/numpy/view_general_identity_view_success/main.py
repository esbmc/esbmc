import numpy as np

i = np.identity(2)
d = i.diagonal()
assert d[0] == 1
assert d[1] == 1
r = i[0]
r[1] = 3
assert i[0][1] == 3
