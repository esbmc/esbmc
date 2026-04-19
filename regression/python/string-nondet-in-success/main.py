
s = nondet_string(11)
assume(s == "hello world")
assert "world" in s
assert "hello" in s
assert "xyz" not in s
