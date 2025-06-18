import numpy as np

# Silent underflow: int8 cannot represent -129
assert np.power(-2, 7, dtype=np.int8) == -128

# Overflow in unsigned type
assert np.power(2, 8, dtype=np.uint8) == 0  # 256 wraps to 0

# Valid case: no overflow
assert np.power(2, 6, dtype=np.int8) == 64
