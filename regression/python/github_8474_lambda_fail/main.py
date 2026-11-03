# An immediately invoked lambda at module level (#8474).
x = nondet_int()
__ESBMC_assume(x > -100 and x < 100)
r = (lambda y: y + 1)(x)
assert r == x + 2
