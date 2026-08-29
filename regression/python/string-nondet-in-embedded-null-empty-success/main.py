s = nondet_str()
__ESBMC_assume(s == "a\0b")

assert "" in s
assert "a\0b" in s
