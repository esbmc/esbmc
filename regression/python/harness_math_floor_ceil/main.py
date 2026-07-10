# Verification harness for math.floor / math.ceil
# (src/python-frontend/models/math.py).
#
# floor(x) is the greatest integer <= x; ceil(x) is the least integer >= x.
# Both reject non-finite inputs (the model asserts not isinf / not isnan).
#
# REQUIRES:
#   R1: x is a finite nondet float, established by bounding it to a real
#       interval.  This is exactly the finiteness precondition the model's
#       internal guards require (see harness_math_floor_nan_fail for the
#       falsification when it is dropped).
#
# ENSURES (f = floor(x), c = ceil(x)):
#   E1: f <= x                                   [floor is a lower bound]
#   E2: x < f + 1                                [floor is the *greatest* one]
#   E3: c >= x                                   [ceil is an upper bound]
#   E4: x > c - 1                                [ceil is the *least* one]
#   E5: f <= c                                   [floor never exceeds ceil]
#   E6: c - f in {0, 1}                          [equal iff x is integral]
import math

x: float = nondet_float()

__ESBMC_assume(x >= -1000.0)
__ESBMC_assume(x <= 1000.0)

f: int = math.floor(x)
c: int = math.ceil(x)

assert f <= x                       # E1
assert x < f + 1                    # E2
assert c >= x                       # E3
assert x > c - 1                    # E4
assert f <= c                       # E5
assert c - f == 0 or c - f == 1     # E6
