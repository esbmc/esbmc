import numpy as np

e = np.eye(2, dtype="int32")
i = np.identity(3, dtype=np.bool_)
assert e[1][1] == 1 and e[0][1] == 0
assert i[2][2] and not i[0][2]
