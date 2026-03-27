# Test: Single element join - wrong assertion (should fail)
l: list[str] = ["hello"]
s = "-".join(l)
assert s == "-hello-"
