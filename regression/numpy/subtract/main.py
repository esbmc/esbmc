import numpy as np

result_1 = np.subtract(10, 4)
assert result_1 == 6

result_2 = np.subtract(5.5, 2.5)
assert result_2 == 3.0

result_3 = np.subtract(7, 2.5)
assert result_3 == 4.5

result_4 = np.subtract(6.8, 2)
assert result_4 == 4.8

result_5 = np.subtract(-3, -7)
assert result_5 == 4

result_6 = np.subtract(-5.5, 2.5)
assert result_6 == -8.0

result_7 = np.subtract(5, -3)
assert result_7 == 8

result_8 = np.subtract(-2.0, 4.0)
assert result_8 == -6.0

result_9 = np.subtract(0, 5)
assert result_9 == -5

result_10 = np.subtract(3.2, 0)
assert result_10 == 3.2
