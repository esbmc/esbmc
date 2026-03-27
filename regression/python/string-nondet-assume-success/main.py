
s = nondet_string(5)
assume(s == "hello")
assert s == "hello"
assert len(s) == 5
