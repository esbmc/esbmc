# Test: String ljust with custom fill
text = "x"
result = text.ljust(3, "-")
assert result == "x--"
assert result != "-x-"
