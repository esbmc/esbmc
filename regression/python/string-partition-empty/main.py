# Test: String partition on empty string
text = ""
parts = text.partition("=")
assert parts == ("", "", "")
assert parts != ("", "=", "")
