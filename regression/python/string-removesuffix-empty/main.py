# Test: String removesuffix on empty string
text = ""
result = text.removesuffix("suffix")
assert result == ""
assert result != "suffix"
