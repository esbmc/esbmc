# Test: String rjust with width smaller than length
text = "abc"
result = text.rjust(2)
assert result == "abc"
assert result != " abc"
