import numpy as np

a = np.add(7, 0, dtype=np.int8)
b = np.add(3, 0, dtype=np.int8)
assert a % b == 1
assert a // b == 2

c = np.add(-7, 0, dtype=np.int8)
assert c // b == -3
assert c % b == 2
