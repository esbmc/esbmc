import numpy as np

e = np.eye(3, k=1, dtype=np.int32)
i = np.identity(2, dtype="float32")
assert e[0][1] == 1 and e[1][1] == 0
assert i[1][1] == 1.0
