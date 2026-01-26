# Test: String format with multiple arguments
text = "{} {} {}"
result = text.format(1, "b", False)
assert result == "1 b False"
assert result != "1 b True"
