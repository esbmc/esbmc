# The set-relation methods work on a set variable receiver (the common form);
# only an inline set-literal receiver is broken (see set_issubset_knownbug).
a = {1, 2}
b = {1, 2, 3}
c = {5, 6}
assert a.issubset(b)
assert not b.issubset(a)
assert b.issuperset(a)
assert a.isdisjoint(c)
assert not a.isdisjoint(b)
