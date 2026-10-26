# Names bound to one str name the same object, also through a chain (#8412).
v = "hello"
b = v
c = b
w = "world"

assert b is v
assert c is b
assert b is not w
