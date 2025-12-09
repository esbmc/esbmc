import numpy as np

y = np.power(3, 21, dtype=np.int32) # 3^21 = 10460353203 -> overflow: 1870418611

assert y == 10460353203