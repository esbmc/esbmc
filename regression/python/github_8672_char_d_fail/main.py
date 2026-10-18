# A str subscript is a one-character str, not an int: "%d" raises TypeError
# in Python, so it is not lowered through %d's int() and stays refused.
s = "abc"
i = 1
assert "<%d>" % s[i] == "<98>"
