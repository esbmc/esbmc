# Test: String partition simple success case
text = "key=value"
parts = text.partition("=")
assert parts == ("key", "=", "value")
