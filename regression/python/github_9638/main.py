# github issue 9638
a = "abcd"
d = a.isdecimal()
assert not d
b = "0123"
assert b.isdecimal()
e = ""
assert not e.isdecimal()
assert not e.istitle()
f = "1.5"
assert not f.isdecimal()
g = "-1"
assert not g.isdecimal()
h = " 1"
assert not h.isdecimal()
t = "It'S A 3D Movie"
x = t.istitle()
assert x
u = "Hello world"
assert not u.istitle()
v = "HELLO"
assert not v.istitle()
