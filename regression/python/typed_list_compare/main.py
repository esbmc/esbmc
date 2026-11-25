def build(n: int) -> list[int]:
    out = []
    i = 0
    while i < n:
        out.append(n - i)
        i += 1
    return out


a = [3, 1, 2]
b = [3, 1, 2]
c = [3, 1]
d = [3, 1, 4]
e = []
assert a == b
assert not (a != b)
assert a != c
assert c != a
assert a != d
assert e == build(0)
assert build(3) == sorted(a, reverse=True)
assert sorted(a) != build(3)
