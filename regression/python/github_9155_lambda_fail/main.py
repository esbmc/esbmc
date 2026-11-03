# Issue #9155: a lambda converts the value the name holds when it is called.
c = True
if c:
    r = 1
else:
    r = "b"
f = lambda: int(r)
r = 2
assert f() == 1
