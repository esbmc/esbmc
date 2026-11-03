# Each overflow below raises OverflowError, so every branch fails (#9392).
import math
from math import expm1

a = 1000.0
r1 = False
r2 = False
r3 = False
r4 = False
try:
    math.expm1(a)
except OverflowError:
    r1 = True
try:
    expm1(a)
except OverflowError:
    r2 = True
try:
    math.ldexp(a, 2000)
except OverflowError:
    r3 = True
try:
    math.exp(1000.0)
except OverflowError:
    r4 = True
k = nondet_int()
if k == 1:
    assert not r1
elif k == 2:
    assert not r2
elif k == 3:
    assert not r3
else:
    assert not r4
