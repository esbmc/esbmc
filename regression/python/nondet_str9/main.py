a = nondet_str()
b = nondet_str()
# This assertion should FAIL because a and b are independent nondeterministic values
# They could be different strings
assert a == b
