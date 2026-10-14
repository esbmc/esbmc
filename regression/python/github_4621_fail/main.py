# Negative test: str(int) must produce the *correct* string, not just any string.
x = 5
s = str(x)
assert s == "6"
