import math

# Very small fractional values
assert math.floor(0.0001) == 0
assert math.ceil(0.0001) == 1
assert math.floor(-0.0001) == -1
assert math.ceil(-0.0001) == 0

# Large values
assert math.floor(1e12 + 0.9) == 1000000000000
assert math.ceil(1e12 + 0.1) == 1000000000001
assert math.floor(-1e12 - 0.1) == -1000000000001
assert math.ceil(-1e12 - 0.9) == -1000000000000
