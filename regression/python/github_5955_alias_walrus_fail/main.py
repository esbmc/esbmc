# The walrus binds m as an alias of l outside any assignment statement; the
# append invalidates the seed (GitHub #5955 review F2).
l = [1, 2]
if (m := l):
    m.append(3)
assert l.count(3) == 0
