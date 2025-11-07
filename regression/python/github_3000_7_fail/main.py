# Test: Four element join - wrong assertion (should fail)
l: list[str] = ["a", "b", "c", "d"]
s = ",".join(l)
assert s == "a,b,c"

