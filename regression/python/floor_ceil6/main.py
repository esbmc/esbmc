# regression/python/floor_ceil_boundary_above/main.py
import math

# Just above 3
x = 3.0000000000001
assert math.floor(x) == 3
assert math.ceil(x) == 4

# Just above -2
y = -1.9999999999999
assert math.floor(y) == -2
assert math.ceil(y) == -1
