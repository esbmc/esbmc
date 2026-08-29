
s = nondet_str()
__ESBMC_assume(s == "hello")
assert s == "hello"
assert len(s) == 5
