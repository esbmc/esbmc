# Test: String zfill on empty string
text = ""
result = text.zfill(3)
assert result == "000"
assert result != ""
