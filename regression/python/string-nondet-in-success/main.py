
s = nondet_str()
__ESBMC_assume(s == "hello world")
assert "world" in s
assert "hello" in s
assert "xyz" not in s
