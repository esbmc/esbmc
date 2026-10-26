import numpy as np

b = np.array([300, 5])
for x in np.nditer(b, op_dtypes=["int8"], flags=["buffered"], casting="same_kind"):
    assert x == 300 or x == 5
