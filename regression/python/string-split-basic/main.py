# Test: String split simples
text = "a,b,c"
parts = text.split(",")
assert len(parts) == 3
assert parts[0] == "a"
assert parts[1] == "b"
assert parts[2] == "c"
