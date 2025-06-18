# Integers
assert abs(-2) == 2
assert abs(2) == 2
assert abs(0) == 0
assert abs(-999999) == 999999
assert abs(999999) == 999999

# Floats
assert abs(-3.5) == 3.5
assert abs(3.5) == 3.5
assert abs(-0.0) == 0.0
assert abs(0.0) == 0.0

# Variables (symbolic)
x = -2.5
assert abs(x) == 2.5

y = 1.000001
assert abs(y) == 1.000001

assert x != y

# Abs on values from another variable
def compute():
    return -7

val = compute()
assert abs(val) == 7

# Abs on expression
z:int = -3 + 1
assert abs(z) == 2
