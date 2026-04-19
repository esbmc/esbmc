# Test: String removeprefix with empty prefix
text = "sample"
result = text.removeprefix("")
assert result == "sample"
assert result != ""
