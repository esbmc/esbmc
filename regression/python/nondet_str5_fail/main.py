# Test nondet_str() - fail case
# Assertion should fail because nondet_str() can return empty string

foo = nondet_str()

# This assertion will fail because nondet_str() can return ""
assert foo != ""

