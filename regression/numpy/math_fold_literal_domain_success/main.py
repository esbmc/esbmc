import math

caught = 0
try:
    math.asin(2.0)
except ValueError:
    caught += 1
try:
    math.atanh(1.0)
except ValueError:
    caught += 1
try:
    math.atanh(-1.0)
except ValueError:
    caught += 1
try:
    math.log1p(-1.0)
except ValueError:
    caught += 1
try:
    math.acosh(0.5)
except ValueError:
    caught += 1
try:
    math.log2(0.0)
except ValueError:
    caught += 1
try:
    math.log10(-2.0)
except ValueError:
    caught += 1
assert caught == 7
