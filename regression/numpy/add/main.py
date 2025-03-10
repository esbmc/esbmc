import numpy as np

result_1 = np.add(1.0, 4.0)
assert result_1 == 5.0

result_2 = np.add(1, 2)
assert result_2 == 3

result_3 = np.add(1, 2.5)
assert result_3 == 3.5

result_4 = np.add(2.5, 1)
assert result_4 == 3.5

result_5 = np.add(-1, -4)
assert result_5 == -5

result_6 = np.add(-1.0, -4.0)
assert result_6 == -5.0

result_7 = np.add(-2, -3.5)
assert result_7 == -5.5

result_8 = np.add(-3.5, -2)
assert result_8 == -5.5

result_9 = np.add(1, -2)
assert result_9 == -1 

result_10 = np.add(1.0, -2.00)
assert result_10 == -1.00

result_11 = np.add(-2, 3)
assert result_11 == 1

result_12 = np.add(-2.5, 3.5)
assert result_12 == 1.0

result_13 = np.add(3.0, -1)
assert result_13 == 2.0

result_14 = np.add(0, 5)
assert result_14 == 5

result_15 = np.add(0.0, -5.0)
assert result_15 == -5.0

result_16 = np.add(5, 0)
assert result_16 == 5

result_17 = np.add(-5.0, 0)
assert result_17 == -5.0 

result_18 = np.add(0, 0.0)
assert result_18 == 0.0

result_19 = np.add(127, 1, dtype=np.int8)
assert result_19 == -128