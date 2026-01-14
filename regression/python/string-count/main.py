# Test: String count
text = "hello world hello"
count1 = text.count("hello")
assert count1 == 2
count2 = text.count("l")
assert count2 == 5
count3 = text.count("xyz")
assert count3 == 0
