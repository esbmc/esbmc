# Test: String format with None
text = "{}"
result = text.format(None)
assert result == "None"
assert result != "null"
