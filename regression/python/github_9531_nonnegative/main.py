x = nondet_int()
__ESBMC_assume(x >= 0)
u = nondet_bool()
d = "s" if u else x
if not u:
    assert d >= 0
