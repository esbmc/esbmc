
s = nondet_str()
__ESBMC_assume(s != "test")
assert s != "test"
# s pode ser qualquer coisa menos "test"
