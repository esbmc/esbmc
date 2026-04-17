# Test: String rjust with custom fill
text = "x"
result = text.rjust(3, "-")
assert result == "--x"
assert result != "-x-"
