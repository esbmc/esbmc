xs: list[int] = [1, 2, 0, 4]
i: int = 0
while not (i < len(xs) and xs[i] == 0):
    i += 1
assert i == 2
