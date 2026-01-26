# Test: String zfill with negative sign
text = "-7"
result = text.zfill(3)
assert result == "-07"
assert result != "0-7"
