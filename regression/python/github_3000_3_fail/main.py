# Test: Empty list join - wrong assertion (should fail)
l: list[str] = []
s = " ".join(l)
assert s == "wrong"

