# Test: str.join() with mixed string and integer types (should fail)
l = ["foo", 123, "bar"]
s = " ".join(l)
assert s == "foo 123 bar"


