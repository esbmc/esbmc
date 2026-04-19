# Test: Empty separator join
l: list[str] = ["a", "b", "c"]
s = "".join(l)
assert s == "abc"

