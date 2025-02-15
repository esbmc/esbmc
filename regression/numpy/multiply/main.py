import numpy as np

result_1 = np.multiply(2.0, 4.0)
assert result_1 == 8.0

result_2 = np.multiply(3, 5)
assert result_2 == 15

result_3 = np.multiply(2.5, 4.0)
assert result_3 == 10.0

result_4 = np.multiply(6, 0)
assert result_4 == 0

result_5 = np.multiply(0, 0)
assert result_5 == 0

result_6 = np.multiply(-3, 4)
assert result_6 == -12

result_7 = np.multiply(-2.0, -5.0)
assert result_7 == 10.0

result_8 = np.multiply(-7, 3.5)
assert result_8 == -24.5

result_9 = np.multiply(-1, -1)
assert result_9 == 1

result_10 = np.multiply(1, 1)
assert result_10 == 1

result_11 = np.multiply(9, -2)
assert result_11 == -18
