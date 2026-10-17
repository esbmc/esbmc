# A str subscript is a one-character str, not an int, so it is not lowered
# through %d's int() and stays refused.
s = "abc"
i = 1
assert "<%s>" % s[i] == "<b>"
