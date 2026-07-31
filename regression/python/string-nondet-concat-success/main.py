
s = nondet_str()
__ESBMC_assume(s == "abc")
result = s + "def"
assert result == "abcdef"
assert len(result) == 6
