import numpy as np

result1 = np.power(2, 3)
assert result1 == 8

result2= np.power(2, -1)
assert result2 == 0.5

result3 = np.power(-2, 3)
assert result3 == -8

y = np.power(2,7, dtype=np.int8)
assert y == -128

# Using dtype int8 and testing overflow behavior
assert np.power(2, 6, dtype=np.int8) == 64
assert np.power(2, 5, dtype=np.int8) == 32

# Basic powers with integers
assert np.power(2, 3) == 8
assert np.power(5, 0) == 1
assert np.power(10, 1) == 10
assert np.power(3, 4) == 81

