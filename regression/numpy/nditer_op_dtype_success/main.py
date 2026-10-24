import numpy as np

f = np.array([1.5, 2.5, -1.5])
truncated = 0
for x in np.nditer(f, op_dtypes=["int64"], flags=["buffered"], casting="unsafe"):
    truncated = truncated * 10 + x
assert truncated == 119
