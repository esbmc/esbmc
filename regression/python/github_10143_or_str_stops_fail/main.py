# #10143: an or stops at a str only when it is non-empty.
t = "a" or 0
assert t == 0
