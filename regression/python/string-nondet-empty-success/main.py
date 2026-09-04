s = nondet_str()
__ESBMC_assume(len(s) == 0)
assert s == ""
result = s + "test"
assert result == "test"
