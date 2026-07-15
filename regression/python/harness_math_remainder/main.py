# Verification harness for math.remainder (src/python-frontend/models/math.py).
#
# remainder(x, y) is the IEEE 754 remainder: x - n*y where n is the integer
# nearest to x/y (ties to even).  Its magnitude never exceeds |y|/2.
#
# REQUIRES:
#   R1: x is a finite nondet float (bounded interval); the divisor is the
#       concrete non-zero value 4.0.
#
# ENSURES (r = remainder(x, 4.0)):
#   E1: r >= -2.0                                [r >= -|y|/2]
#   E2: r <=  2.0                                [r <=  |y|/2]
#
# Concrete anchors pin the round-to-nearest behaviour at representative points.
import math

x: float = nondet_float()

__ESBMC_assume(-100.0 <= x <= 100.0)

r: float = math.remainder(x, 4.0)

assert r >= -2.0        # E1
assert r <= 2.0         # E2

assert math.remainder(5.0, 3.0) == -1.0    # nearest multiple of 3 to 5 is 6
assert math.remainder(7.0, 3.0) == 1.0     # nearest multiple of 3 to 7 is 6
assert math.remainder(10.0, 4.0) == 2.0    # tie 8/12 -> even multiple 8
