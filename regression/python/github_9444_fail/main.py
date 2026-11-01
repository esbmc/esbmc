# Issue #9444: round(int, n) with n >= 0 returns the int unchanged, so it can be 15.
x: int = nondet_int()
r: int = round(x, 1)
assert r != 15
