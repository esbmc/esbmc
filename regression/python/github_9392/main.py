# A finite argument whose result overflows raises OverflowError (#9392).
import cmath
import math
from math import expm1, ldexp

a = 1000.0
raised = 0
try:
    math.expm1(a)
except OverflowError:
    raised += 1
try:
    expm1(a)
except OverflowError:
    raised += 1
try:
    math.ldexp(a, 2000)
except OverflowError:
    raised += 1
try:
    ldexp(1.0, 1024)
except OverflowError:
    raised += 1
try:
    math.ldexp(1.0, 2**40)
except OverflowError:
    raised += 1
try:
    math.exp(1000.0)
except OverflowError:
    raised += 1
try:
    math.sinh(1000.0)
except OverflowError:
    raised += 1
try:
    math.cosh(-1000.0)
except OverflowError:
    raised += 1
try:
    math.exp2(1100.0)
except OverflowError:
    raised += 1
assert raised == 9

inf = float("inf")
assert math.expm1(-1000.0) == -1.0
assert math.exp(inf) == inf
assert math.expm1(inf) == inf
assert math.ldexp(0.0, 2000) == 0.0
assert ldexp(1.0, -2000) == 0.0
assert math.ldexp(inf, 5) == inf
assert math.ldexp(3.0, 2) == 12.0
assert math.ldexp(1.0, -2**40) == 0.0
assert math.ldexp(1e-300, 1100) > 1.3e31

# cmath.tanh builds on math.sinh/cosh, but CPython does not overflow there.
tanh_raised = False
try:
    cmath.tanh(complex(1000.0, 0.0))
except OverflowError:
    tanh_raised = True
assert not tanh_raised


# A short-circuited call must not raise.
def guarded(c: bool) -> bool:
    return (c and math.expm1(1000.0) > 0.0) or (c and math.exp(1000.0) > 0.0) or (
        c and math.ldexp(1.0, 5000) > 0.0)


assert not guarded(False)


def taken(c: bool) -> bool:
    try:
        return c and math.cosh(-1000.0) > 0.0
    except OverflowError:
        return False


assert not taken(True)

# CPython keeps these finite; the C models of sinh and cosh overflow earlier.
near = False
try:
    math.sinh(-710.0)
    math.cosh(-710.4758600739439)
    math.expm1(709.782712893384)
    math.exp2(1023.9)
except OverflowError:
    near = True
assert not near
