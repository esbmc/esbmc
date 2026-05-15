import cmath
import math

r, p = cmath.polar(complex(float('nan'), 1.0))
assert r != r
assert p != p
