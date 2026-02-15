# Test: String splitlines on empty string
text = ""
parts = text.splitlines()
assert parts == []
assert parts != [""]
