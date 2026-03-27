import math

# Positive numbers
assert math.floor(3.7) == 3
assert math.ceil(3.7) == 4

# Negative numbers
assert math.floor(-3.7) == -4
assert math.ceil(-3.7) == -3

# Zero
assert math.floor(0.0) == 0
assert math.ceil(0.0) == 0

# Integers (should stay the same)
assert math.floor(5.0) == 5
assert math.ceil(5.0) == 5
assert math.floor(-5.0) == -5
assert math.ceil(-5.0) == -5
