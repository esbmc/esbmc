# Test: String partition with separator at end
text = "value="
parts = text.partition("=")
assert parts == ("value", "=", "")
assert parts != ("value", "", "=")
