# Test: String splitlines with CRLF
text = "line1\r\nline2"
parts = text.splitlines()
assert parts == ["line1", "line2"]
assert parts != ["line1\r", "line2"]
