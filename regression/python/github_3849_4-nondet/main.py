a: int = nondet_int()
b: int = nondet_int()
c: int = nondet_int()

__ESBMC_assume(a >= 0 and a <= 100)
__ESBMC_assume(b >= 0 and b <= 100)
__ESBMC_assume(c >= 0 and c <= 100)

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
