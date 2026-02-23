from esbmc import nondet_int, assume, esbmc_assert

x: int = nondet_int()
assume(x > 0 and x < 100)
