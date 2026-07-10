# Verification harness for math.trunc (src/python-frontend/models/math.py).
#
# trunc(x) rounds x toward zero: floor(x) for x >= 0, ceil(x) for x < 0.
#
# REQUIRES:
#   R1: x is a finite nondet float (bounded interval).
#
# ENSURES (t = trunc(x)):
#   E1: for x >= 0, t == floor(x)               [rounds down toward 0]
#   E2: for x <  0, t == ceil(x)                [rounds up toward 0]
#   E3: abs(t) <= abs(x)                        [truncation never grows the
#       magnitude]
import math

x: float = nondet_float()

__ESBMC_assume(x >= -1000.0)
__ESBMC_assume(x <= 1000.0)

t: int = math.trunc(x)

if x >= 0.0:
    assert t == math.floor(x)       # E1
    assert t <= x                   # E3 (non-negative side)
else:
    assert t == math.ceil(x)        # E2
    assert t >= x                   # E3 (negative side)
