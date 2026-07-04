# list.index(x) now raises a catchable ValueError when x is absent, instead of
# an uncatchable property violation, so try/except ValueError works.
l = [1, 2, 3]
try:
    l.index(5)
    assert False
except ValueError:
    pass

# A present element still returns its (first) index.
assert l.index(2) == 1
assert [1, 1, 2].index(1) == 0
names = ["a", "b", "c"]
assert names.index("b") == 1
