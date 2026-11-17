import math

r = 1.0
try:
    r = math.pow(-8.0, 1.0 / 3.0)
except ValueError:
    pass
assert r == 1.0
try:
    r = math.gamma(-2.0)
except ValueError:
    pass
assert r == 1.0
try:
    r = math.pow(10.0, 400.0)
except OverflowError:
    pass
assert r == 1.0
