# Test: Direct list literal with wrong assertion (would fail if implemented)
# This is a negative test for the TODO feature
s = " ".join(["foo", "bar"])
assert s == "foobar"  # Wrong! Should be "foo bar"
