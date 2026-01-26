# Test: String zfill with width smaller than length
text = "1234"
result = text.zfill(2)
assert result == "1234"
assert result != "01234"
