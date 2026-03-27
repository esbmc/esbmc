s = nondet_string(5)
assume(s == "hello")
upper = s.upper()
assert upper == "HELLO"
assert len(upper) == 5
