import math

# NaN should not be a valid input
x = float("nan")
assert math.floor(x) == 0  # This should FAIL
assert math.ceil(x) == 0   # This should FAIL

