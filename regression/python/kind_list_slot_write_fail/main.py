def bump(xs: list[int]) -> None:
    i = 0
    while i < len(xs):
        if i == 6:
            assert xs[3] == 0
        xs[i] = 7
        i += 1


n = nondet_int()
xs = [0] * n
bump(xs)
