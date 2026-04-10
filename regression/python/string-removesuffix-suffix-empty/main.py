# Test: String removesuffix with empty suffix
text = "sample"
result = text.removesuffix("")
assert result == "sample"
assert result != ""
