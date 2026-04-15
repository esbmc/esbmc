xs: list[int] = [1, 2]
assert all(xs) == True

ys: list[int] = [1, 0, 2]
assert all(ys) == False
