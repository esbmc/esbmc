# Test: String format simple success case
text = "{} {}"
result = text.format("a", "b")
assert result == "a b"
