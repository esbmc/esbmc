n = 0


def g() -> int:
    global n
    n += 1
    return n


xs = [g(), g()]
ys = [b for b in xs]
assert ys == [1, 2]
assert n == 4
