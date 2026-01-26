# Test: String format with extra arguments
text = "{}"
result = text.format("a", "b")
assert result == "a"
assert result != "b"
