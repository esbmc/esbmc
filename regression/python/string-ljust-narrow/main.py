# Test: String ljust with width smaller than length
text = "abc"
result = text.ljust(2)
assert result == "abc"
assert result != "abc "
