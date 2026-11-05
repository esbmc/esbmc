# #9821, false assertion: a str read by subscript and stored under another
# key stays a str.
x = {}
x["k"] = "s"
d = x["k"]
x["j"] = d
e = x.pop("j")
assert e[0] == "t"
