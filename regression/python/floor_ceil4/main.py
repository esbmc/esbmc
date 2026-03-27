import math

# Very small positive/negative values
assert math.floor(1e-10) == 0
assert math.ceil(1e-10) == 1
assert math.floor(-1e-10) == -1
assert math.ceil(-1e-10) == 0

# Values near zero
assert math.floor(0.5) == 0
assert math.ceil(0.5) == 1
assert math.floor(-0.5) == -1
assert math.ceil(-0.5) == 0
