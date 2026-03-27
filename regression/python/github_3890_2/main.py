# Test type() for multiple types
x = 0
assert (type(x) == int)
assert (type(x) != str)
assert (type(x) != float)

s = "hello"
assert (type(s) == str)
assert (type(s) != int)

f = 3.14
assert (type(f) == float)
assert (type(f) != int)

b = True
assert (type(b) == bool)
