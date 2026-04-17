# Test: String expandtabs with multiple tabs
text = "a\tb\tc"
result = text.expandtabs(4)
assert result == "a   b   c"
