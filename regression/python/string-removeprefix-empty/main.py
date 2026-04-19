# Test: String removeprefix on empty string
text = ""
result = text.removeprefix("prefix")
assert result == ""
assert result != "prefix"
