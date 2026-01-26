# Test: String expandtabs without tabs
text = "sample"
result = text.expandtabs(4)
assert result == "sample"
