l = [10, 20, 30, 40, 50]
s = nondet_int()
__ESBMC_assume(s == 2 or s == 3)
r = l[::s]
assert len(r) == 3
