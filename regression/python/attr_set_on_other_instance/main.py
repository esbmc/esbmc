class R:
    pass


a = R()
b = R()
a.prev = 3
try:
    b.prev
    assert False
except AttributeError:
    pass
assert a.prev == 3
