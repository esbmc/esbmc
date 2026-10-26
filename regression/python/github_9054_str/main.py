# GitHub #9054: str.index(), rindex(), count() and find() return an int.
s = "abc"
i = s.index("a")
assert isinstance(i, int)
r = s.rindex("c")
assert isinstance(r, int)
n = s.count("a")
assert isinstance(n, int)
f = s.find("b")
assert isinstance(f, int)
