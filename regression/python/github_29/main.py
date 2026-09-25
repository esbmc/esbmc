# Element types must survive the container operations that forward them:
# reverse() reorders the recorded sequence, pop() drops its tail, and a
# later append() re-types an emptied list.
l = [1, 2.5, 3]
l.reverse()
assert l[0] == 3
assert l[1] == 2.5
assert l[2] == 1

m = [4.5]
m.pop()
m.append(7)
assert m[0] == 7

d = {"a": 1, "b": 2.5}
assert d["a"] == 1
assert d["b"] == 2.5
for v in d.values():
    assert v == 1 or v == 2.5
