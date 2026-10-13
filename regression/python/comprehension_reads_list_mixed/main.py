n = 0


def g() -> int:
    global n
    n += 1
    return n


xs = [g(), None, 5]
ys = [b for b in xs if b is not None]
assert ys == [1, 5]
assert n == 1
