# Test: Single element join should return the element itself
l: list[str] = ["hello"]
s = "-".join(l)
assert s == "hello"
