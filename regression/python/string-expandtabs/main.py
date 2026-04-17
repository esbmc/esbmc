# Test: String expandtabs simple success case
text = "a\tb"
result = text.expandtabs(4)
assert result == "a   b"
