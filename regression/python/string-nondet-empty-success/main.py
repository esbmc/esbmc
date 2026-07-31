
s = nondet_str()
assert len(s) == 0
assert s == ""
result = s + "test"
assert result == "test"
