# Issue #9444: round(x, ndigits) with a non-literal x crashed in GOTO conversion.
# round(int, n) with n >= 0 returns the int unchanged.
x: int = nondet_int()
r: int = round(x, 1)
assert r == x
assert round(x, 0) == x
