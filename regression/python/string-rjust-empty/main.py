# Test: String rjust on empty string
text = ""
result = text.rjust(3)
assert result == "   "
assert result != ""
