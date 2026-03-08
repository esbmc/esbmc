n: int = nondet_int()
__ESBMC_assume(1 <= n <= 5)
r: range = range(n)
it = r.__iter__()
assert it is not None
