import cmath
import math

# atanh boundary exceptions at +/-1
raised = False
try:
    cmath.atanh(complex(1.0, 0.0))
except ValueError:
    raised = True
assert raised

raised = False
try:
    cmath.atanh(complex(-1.0, 0.0))
except ValueError:
    raised = True
assert raised
