# The `else` clause of a try runs when the body completes without an exception,
# and is skipped when an exception is caught. It was previously dropped
# silently. Also covers try/except/else/finally, which was refused before.
x = 0
try:
    x = 1
except:
    x = 2
else:
    x = 3
assert x == 3           # no exception -> else runs

y = 0
try:
    raise ValueError()
except:
    y = 2
else:
    y = 3
assert y == 2           # exception caught -> else skipped

# try/except/else/finally ordering.
r = []
try:
    r.append(1)
except:
    r.append(2)
else:
    r.append(3)
finally:
    r.append(4)
assert r == [1, 3, 4]

# else + finally on the exception path.
s = []
try:
    raise KeyError()
except KeyError:
    s.append(2)
else:
    s.append(3)
finally:
    s.append(4)
assert s == [2, 4]
