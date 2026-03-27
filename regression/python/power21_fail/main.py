import math

# Large exponents leading to overflow
assert math.isinf(10.0 ** 400)
assert (10.0 ** -400) == 0.0

# Small base, large negative exponent
assert math.isinf((1e-100) ** -100)
assert (1e-100) ** 100 == 0.0

# Large base, small exponent (no overflow)
assert abs((1e100) ** 0.01 - 1.5848931924611136) < 1e-12

