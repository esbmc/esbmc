x = dict()
x[2.0] = "string"
a = x.get(2.0)
if a == None:
    a = ""
d = x.pop(2.0)
x[2.0] = d
e = x.popitem()
assert a[0] + d[0] == "ss"
assert isinstance(e[0], float)
assert e[1] == "string"
