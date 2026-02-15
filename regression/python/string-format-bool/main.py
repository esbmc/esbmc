# Test: String format with boolean
text = "{}"
result = text.format(True)
assert result == "True"
assert result != "true"
