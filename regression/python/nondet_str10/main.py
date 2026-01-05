s = nondet_str()
original = s
s = "modified"
# This assertion should FAIL because s was reassigned
assert s == original
