s = "foo"
b: bool = "f" in s
assert b

b = "l" in s
assert not b

t = "needle in haystack"
c: bool = "needle" in t
assert c

c = "neddle" in t
assert not c

x = "hello world"
y = "world"
assert y in x
assert not "mars" in x

empty = ""
assert not "x" in empty
assert "" in "abc"
