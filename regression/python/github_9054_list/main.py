# GitHub #9054: list.index() and list.count() return an int.
x = ["a"]
z = x.index("a")
assert isinstance(z, int)
y = [1, 2]
w = y.index(2)
assert isinstance(w, int)
assert isinstance(y.count(2), int)
c = y.count(2)
assert isinstance(c, int)
assert not isinstance(c, str)
