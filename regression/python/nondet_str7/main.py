# Test nondet_str() with string method-like operations

s1 = nondet_str()
s2 = nondet_str()

# Compare two nondeterministic strings
if s1 == s2:
    # If they are equal, this assertion should pass
    assert s1 == s2
else:
    # If they are different, this assertion should pass
    assert s1 != s2

