import numpy as np

a = np.array([1, 2, 3, 4])
part = a[::-1]

assert len(part) == 4
assert part[0] == 4
assert part[3] == 1
