# Test: Multi-character separator - wrong assertion (should fail)
l: list[str] = ["x", "y"]
s = "::".join(l)
assert s == "x:y"

