def same(a: list[int], b: list[int]) -> bool:
    if len(a) != len(b):
        return False
    i = 0
    while i < len(a):
        if a[i] != b[i]:
            return False
        i += 1
    return True


l = [10, 20, 30, 40, 50]
assert same(l[1:3], [20, 30])
assert same(l[:2], [10, 20])
assert same(l[3:], [40, 50])
assert same(l[-2:], [40, 50])
assert same(l[:-3], [10, 20])
assert same(l[:len(l) - 1], [10, 20, 30, 40])
assert same(l[-100:100], [10, 20, 30, 40, 50])
assert same(l[4:1], [])
assert same(l[::2], [10, 30, 50])
assert same(l[1::3], [20, 50])
assert same(l[::-1], [50, 40, 30, 20, 10])
assert same(l[3:0:-2], [40, 20])
assert same(l[-1:-4:-1], [50, 40, 30])
assert same(l[:-6:-2], [50, 30, 10])
assert same(l[10:-10:-3], [50, 20])
assert same(l[0:5:-1], [])
assert same(l[::9223372036854775807], [10])

c = list(l)
d = l.copy()
c[0] = 1
d.append(60)
assert l[0] == 10
assert len(l) == 5
assert same(c, [1, 20, 30, 40, 50])
assert same(d, [10, 20, 30, 40, 50, 60])

# Results compared as untyped lists, from the same typed source.
assert l[1:4:2] == [20, 40]
assert l.copy() == [10, 20, 30, 40, 50]
assert list(l)[::-2] == [50, 30, 10]
