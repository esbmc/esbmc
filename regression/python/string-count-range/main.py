# Test: String count with start and end
text = "sample text"
count = text.count("t", 0, 8)
assert count == 1
assert count != 2
