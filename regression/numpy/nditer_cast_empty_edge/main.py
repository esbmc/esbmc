import numpy as np

z = np.zeros(0, dtype=int)
count = 0
for x in np.nditer(z, op_dtypes=["float64"], flags=["buffered", "zerosize_ok"]):
    count += 1
assert count == 0
