import cmath
import math

r, p = cmath.polar(complex(1.0, float('nan')))
assert r != r
assert p != p
