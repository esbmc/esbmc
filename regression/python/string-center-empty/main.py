# Test: String center on empty string
text = ""
result = text.center(3)
assert result == "   "
assert result != ""
