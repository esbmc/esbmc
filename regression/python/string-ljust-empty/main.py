# Test: String ljust on empty string
text = ""
result = text.ljust(3)
assert result == "   "
assert result != ""
