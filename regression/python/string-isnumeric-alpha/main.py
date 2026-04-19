# Test: String isnumeric with alphabetic characters
text = "123a"
result = text.isnumeric()
assert result is False
