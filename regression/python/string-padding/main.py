# Test: String zfill e ljust/rjust
num = "42"
padded = num.zfill(5)
assert padded == "00042"
text = "hi"
left = text.ljust(5)
assert left == "hi   "
right = text.rjust(5)
assert right == "   hi"
