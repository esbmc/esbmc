import numpy as np

assert np.power(2, 7, dtype=np.int8) == -128  # 128 overflows in int8
