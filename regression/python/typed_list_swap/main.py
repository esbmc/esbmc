def same(a: list[int], b: list[int]) -> bool:
    if len(a) != len(b):
        return False
    i = 0
    while i < len(a):
        if a[i] != b[i]:
            return False
        i += 1
    return True


l = [10, 20, 30]
i = 0
j = 2
l[i], l[j] = l[j], l[i]
assert same(l, [30, 20, 10])
x, l[1] = l[1], 5
assert x == 20 and same(l, [30, 5, 10])
