# #9377: an empty str operand of and/or is false.
s = ""
t = "a"
assert (t or "x") == "a"
assert (s or "x") != "x"
