# github issue 9438: str(x, encoding) of a name bound to None is not "None"
x = None
s = str(x, "utf-8")
assert s == "None"
