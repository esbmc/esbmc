# Test: String expandtabs with default tabsize
text = "a\tb"
result = text.expandtabs()
assert result == "a       b"
