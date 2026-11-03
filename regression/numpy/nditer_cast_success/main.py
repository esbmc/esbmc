import numpy as np

a = np.array([1, 2, 3])
total = 0.0
for x in np.nditer(a, op_dtypes=["float64"], flags=["buffered"]):
    total += x / 4
assert total == 1.5
