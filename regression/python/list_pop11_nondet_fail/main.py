x = nondet_int()
__ESBMC_assume(x >= 0 and x <= 10)

y = nondet_float()
__ESBMC_assume(y >=0 and y <= 10.0)

l = [x, y]

assert l.pop() + l.pop() <= 19
