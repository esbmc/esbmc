import numpy as np

a = np.array([1, 2, 3, 4])
part = a[::2]

assert len(part) == 2
assert part[0] == 1
assert part[1] == 3
