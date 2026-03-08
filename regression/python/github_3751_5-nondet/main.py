step: int = nondet_int()
__ESBMC_assume(step == 1 or step == 2)
r: range = range(0, 6, step)
it = r.__iter__()
assert it is not None
