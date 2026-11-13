import math

x = nondet_float()
caught = False
if x == -3.0:
    try:
        math.gamma(x)
    except ValueError:
        caught = True
    assert caught
y = nondet_float()
if y >= 172.0 and y <= 200.0:
    raised = False
    try:
        math.gamma(y)
    except OverflowError:
        raised = True
    assert raised
tiny = False
try:
    math.gamma(-1e-320)
except OverflowError:
    tiny = True
assert tiny
