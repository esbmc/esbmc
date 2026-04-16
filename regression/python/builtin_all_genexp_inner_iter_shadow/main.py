xs: list[int] = [2, 3]

assert all(x > 0 for x in xs for x in range(x)) == False
