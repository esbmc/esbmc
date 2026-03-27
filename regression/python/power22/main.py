import math

nan = float('nan')

# NaN base or exponent propagates
assert math.isnan(nan ** 2)
assert math.isnan(2 ** nan)
assert math.isnan(nan ** nan)

# But 1 ** NaN is defined as 1.0
assert (1.0 ** nan) == 1.0

