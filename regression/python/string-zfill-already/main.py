# Test: String zfill already padded
text = "007"
result = text.zfill(3)
assert result == "007"
assert result != "0007"
