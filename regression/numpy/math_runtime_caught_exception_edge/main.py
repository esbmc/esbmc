import math

x = nondet_float()
caught = 0
if x > 710.48 and x < float("inf"):
    try:
        y = math.cosh(x)
    except OverflowError:
        caught = 1
    assert caught == 1
if x > 1.0:
    try:
        y = math.acos(x)
    except ValueError:
        caught = 2
    assert caught == 2
