# scalar % ndarray broadcasting is not modelled; ESBMC must reject it with a
# clean diagnostic rather than crashing (#5498).
import numpy as np

a = np.array([4, 5, 6])
b = 10 % a
assert b[0] == 2
