# github issue 9638
a = "abcd"
d = a.isdecimal()
assert not d
b = "0123"
assert b.isdecimal()
e = ""
assert not e.isdecimal()
assert not e.istitle()
t = "It'S A 3D Movie"
x = t.istitle()
assert x
u = "Hello world"
assert not u.istitle()
v = "HELLO"
assert not v.istitle()
