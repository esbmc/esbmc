# GitHub #9054: methods that return an element or a list keep their type.
x = ["a"]
e = x.pop()
assert isinstance(e, str)
y = [1, 2]
c = y.copy()
assert isinstance(c, list)
assert not isinstance(c, int)
