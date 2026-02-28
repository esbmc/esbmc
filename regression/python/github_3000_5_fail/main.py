# Test: Empty separator join - wrong assertion (should fail)
l: list[str] = ["a", "b", "c"]
s = "".join(l)
assert s == "a b c"
