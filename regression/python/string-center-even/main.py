# Test: String center with even padding
text = "ab"
result = text.center(4)
assert result == " ab "
assert result != "  ab"
