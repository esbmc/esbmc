# Test: String count with absent substring
text = "sample text"
count = text.count("z")
assert count == 0
assert count != 1
