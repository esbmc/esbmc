s = nondet_string(3)
assume(s == "abc")
result = s + "def"
assert result == "abcdef"
assert len(result) == 6
