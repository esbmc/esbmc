import math

x = nondet_float()
y = nondet_float()
assert math.pow(x, 0.0) == 1.0
assert math.pow(1.0, y) == 1.0
if x >= -3.0 and x <= -2.0:
    assert math.pow(x, 3.0) < 0.0
    assert math.pow(x, 2.0) > 0.0
caught = 0
try:
    math.pow(0.0, -1.0)
except ValueError:
    caught += 1
try:
    math.pow(10.0, 400.0)
except OverflowError:
    caught += 1
assert caught == 2
