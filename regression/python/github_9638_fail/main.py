# github issue 9638
b = "0123"
t = "Hello World"
d = b.isdecimal()
assert d
assert not t.istitle()
