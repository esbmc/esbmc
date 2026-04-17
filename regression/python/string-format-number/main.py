# Test: String format with number
text = "{}"
result = text.format(42)
assert result == "42"
assert result != "0042"
