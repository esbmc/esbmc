x = nondet_str()
# This assertion should FAIL because x is nondeterministic
# It could be any string (including empty), not necessarily "hello"
assert x == "hello"
