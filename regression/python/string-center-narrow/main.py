# Test: String center with width smaller than length
text = "abc"
result = text.center(2)
assert result == "abc"
assert result != " ab "
