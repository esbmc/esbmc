d = 5


def bump(xs: list[int]) -> None:
    i = 0
    while i < len(xs):
        xs[i] = 10 // d
        i += 1


n = nondet_int()
xs = [0] * n
bump(xs)
