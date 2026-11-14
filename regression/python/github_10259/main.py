def total(xs: list[int]) -> int:
    s: int = 0
    for x in xs:
        s = s + x
    return s


ys = [4, 5]
n = 0
for y in ys:
    n = n + y

assert total([1, 2, 3]) == 6
assert n == 9
