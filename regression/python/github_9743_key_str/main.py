x = dict()
x["a"] = 1
e = x.popitem()
assert e[0] == "a"
assert e[1] == 1
