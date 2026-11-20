# #10143: an or stops at a str only when it is non-empty.
s = "" or None
assert s is None
t = "a" or 0
assert t == "a"
