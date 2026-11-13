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
l[1:3] = [7]
assert same(l, [10, 7, 40, 50])
l = [10, 20, 30, 40, 50]
l[1:2] = [7, 8, 9]
assert same(l, [10, 7, 8, 9, 30, 40, 50])
l = [10, 20, 30]
l[3:] = [40]
assert same(l, [10, 20, 30, 40])
l = [10, 20, 30]
l[:] = [1]
assert same(l, [1])
l = [10, 20, 30]
l[2:1] = [5]
assert same(l, [10, 20, 5, 30])
l = [10, 20, 30, 40]
l[-2:] = []
assert same(l, [10, 20])
l = [10, 20, 30, 40]
l[::2] = [1, 3]
assert same(l, [1, 20, 3, 40])
l = [10, 20, 30, 40]
l[::-1] = [1, 2, 3, 4]
assert same(l, [4, 3, 2, 1])
l = [10, 20, 30]
l[1:] = l
assert same(l, [10, 10, 20, 30])
l = [10, 20, 30, 40]
l[1:] = reversed(l[1:])
assert same(l, [10, 40, 30, 20])
m = [1, 2, 3]
l = [10, 20, 30]
l[:2] = reversed(m)
assert same(l, [3, 2, 1, 30])
assert same(m, [1, 2, 3])
l = [10, 20, 30, 40]
l[0:2] = m[1:]
assert same(l, [2, 3, 30, 40])
