# Issue #9444: round(float, ndigits) of a non-literal is not modelled; it must be
# refused as a failed check instead of crashing.
x: float = 15.0
r: float = round(x, -1)
assert r == 20.0
