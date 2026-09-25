
s = nondet_str()
__ESBMC_assume(s == "hello")
upper = s.upper()
assert upper == "HELLO"
assert len(upper) == 5
