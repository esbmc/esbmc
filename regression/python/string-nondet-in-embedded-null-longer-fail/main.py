s = nondet_str()
__ESBMC_assume(s == "a\0b")

assert "a\0bc" in s
