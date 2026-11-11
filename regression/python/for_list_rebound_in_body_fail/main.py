xs: list[int] = [1, 2, 3]
n: int = 0
for x in xs:
    xs = []
    n = n + 1
assert n == 1
