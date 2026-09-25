
s1 = nondet_str()
s2 = nondet_str()
__ESBMC_assume(s1 == "abc")
__ESBMC_assume(s2 == "abc")
assert s1 == s2
assert s1 == "abc"
