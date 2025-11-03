# Test: Multi-character separator
l: list[str] = ["x", "y"]
s = "::".join(l)
assert s == "x::y"

