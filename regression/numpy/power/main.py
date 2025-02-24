import numpy as np

result1 = np.power(2, 3)
assert result1 == 8

result2= np.power(2, -1)
assert result2 == 0.5

result3 = np.power(-2, 3)
assert result3 == -8

result4 = np.power(2,7, dtype=np.int8)
assert result4 == -128