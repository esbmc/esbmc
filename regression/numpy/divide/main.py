import numpy as np

result_1 = np.divide(2.0, 4.0)
assert result_1 == 0.5

result_2 = np.divide(4.0, 2.0)
assert result_2 == 2.0

result_3 = np.divide(10, 5)
assert result_3 == 2.0

result_4 = np.divide(7.5, 2.5)
assert result_4 == 3.0

result_5 = np.divide(-6, 3)
assert result_5 == -2.0

result_6 = np.divide(6, -3)
assert result_6 == -2.0

result_7 = np.divide(-8, -2)
assert result_7 == 4.0

result_8 = np.divide(0, 5)
assert result_8 == 0.0

result_9 = np.divide(5, 1)
assert result_9 == 5.0

result_10 = np.divide(-10.0, 2.0)
assert result_10 == -5.0

result_11 = np.divide(-50, -2)
assert result_11 == 25.0

result_12 = np.divide(7.0, -3.5)
assert result_12 == -2.0
