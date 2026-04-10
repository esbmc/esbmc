# Test: String removeprefix simple success case
text = "prefix_value"
result = text.removeprefix("prefix_")
assert result == "value"
