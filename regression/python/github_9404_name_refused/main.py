# #9404: a None held in a variable is refused, not treated as arithmetic.
x = None
s = "%s" % x
assert s == "None"
