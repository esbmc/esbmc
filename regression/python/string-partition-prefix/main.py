# Test: String partition with separator at start
text = "=value"
parts = text.partition("=")
assert parts == ("", "=", "value")
assert parts != ("=", "value", "")
