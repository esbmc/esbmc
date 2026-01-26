# Test: String format with escaped braces
text = "{{}}"
result = text.format()
assert result == "{}"
