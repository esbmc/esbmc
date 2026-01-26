# Test: String format with indexed argument
text = "{1} {0}"
result = text.format("a", "b")
assert result == "b a"
assert result != "a b"
