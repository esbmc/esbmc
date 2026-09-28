import numpy as np

a = np.array([1, 2, 3, 4])
part = a[1:3]

assert len(part) == 2
assert part[0] == 2
assert part[1] == 3
