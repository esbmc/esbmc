r = "a"
t = type(r)
assert t == str
assert t is not int
assert type(r) == type("x")
assert str(type(r)) == "<class 'str'>"
n = 5
assert str(type(n)) == "<class 'int'>"
assert str(type(n)) != "<class 'str'>"
