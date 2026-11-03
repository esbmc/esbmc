# #9401: float() strips only leading and trailing whitespace.
import math

assert float(" 2.5 ") == 2.5
assert float("2.5\n") == 2.5
assert math.isinf(float("\t-inf "))
try:
    x = float("in f")
except ValueError:
    x = 0.0
assert x == 0.0
try:
    y = float("  ")
except ValueError:
    y = 1.0
assert y == 1.0
