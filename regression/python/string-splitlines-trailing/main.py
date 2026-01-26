# Test: String splitlines with trailing newline
text = "line1\nline2\n"
parts = text.splitlines()
assert parts == ["line1", "line2"]
assert parts != ["line1", "line2", ""]
