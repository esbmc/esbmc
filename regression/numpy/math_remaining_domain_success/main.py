import math

# math.* raises where numpy.* returns NaN or an infinity.
x = nondet_float()
raised = False
try:
    if x <= -1.0:
        math.log1p(x)
    elif x < 1.0 and x > 0.0:
        math.acosh(x)
    elif x >= 1.0 and x < 2.0:
        math.atanh(x)
    else:
        raised = True
except ValueError:
    raised = True
assert raised or x != x

overflowed = 0
try:
    math.expm1(710.0)
except OverflowError:
    overflowed += 1
try:
    math.exp2(1024.0)
except OverflowError:
    overflowed += 1
assert overflowed == 2
