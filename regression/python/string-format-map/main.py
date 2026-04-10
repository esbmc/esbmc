# Test: String format_map with constant dict
text = "{x}"
result = text.format_map({"x": 1})
assert result == "1"
assert result != "0"
