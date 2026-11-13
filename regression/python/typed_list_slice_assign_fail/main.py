def same(a: list[int], b: list[int]) -> bool:
    if len(a) != len(b):
        return False
    i = 0
    while i < len(a):
        if a[i] != b[i]:
            return False
        i += 1
    return True


l = [10, 20, 30, 40]
l[1:] = reversed(l[1:])
assert same(l, [10, 20, 30, 40])
