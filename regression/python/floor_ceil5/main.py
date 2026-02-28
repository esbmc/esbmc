import math

# Just below 3
x = 2.9999999999999
assert math.floor(x) == 2
assert math.ceil(x) == 3

# Just below -2
y = -2.0000000000001
assert math.floor(y) == -3
assert math.ceil(y) == -2
