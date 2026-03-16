a: float = nondet_float()
b: float = nondet_float()
c: float = nondet_float()

__ESBMC_assume(a >= 0.0 and a <= 10.0)
__ESBMC_assume(b >= 0.0 and b <= 10.0)
__ESBMC_assume(c >= 0.0 and c <= 10.0)

m = min(a, b, c)

# Result must be <= every argument
assert m <= a
assert m <= b
assert m <= c

# Result must equal one of the arguments
assert m == a or m == b or m == c

mx = max(a, b, c)

# Result must be >= every argument
assert mx >= a
assert mx >= b
assert mx >= c

# Result must equal one of the arguments
assert mx == a or mx == b or mx == c
