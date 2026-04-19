# Test: String partition with absent separator
text = "sample text"
parts = text.partition("=")
assert parts == ("sample text", "", "")
assert parts != ("sample", "=", "text")
