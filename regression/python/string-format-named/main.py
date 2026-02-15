# Test: String format with named argument
text = "{name}"
result = text.format(name="a")
assert result == "a"
assert result != "b"
