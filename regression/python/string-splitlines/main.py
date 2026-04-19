# Test: String splitlines simple success case
text = "line1\nline2"
parts = text.splitlines()
assert parts == ["line1", "line2"]
