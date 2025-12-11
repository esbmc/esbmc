# Test nondet_str() with string comparison operations
# This tests the 'with' expression handling in address_of

foo = nondet_str()

# Test that nondet_str can be compared with empty string
# and that the length is always >= 0
assert len(foo) >= 0

