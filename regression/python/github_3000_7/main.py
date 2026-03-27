# Test: Four element join
l: list[str] = ["a", "b", "c", "d"]
s = ",".join(l)
assert s == "a,b,c,d"
