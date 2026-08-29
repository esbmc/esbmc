import numpy as np

# A constant range past the supported element limit must decline explicitly
# and quickly instead of materializing a huge literal list up front.
a = np.arange(1000000)
assert len(a) == 1000000
