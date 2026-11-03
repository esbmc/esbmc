n = 0


def g() -> int:
    global n
    n += 1
    return n


xs = [g(), g(), g()]
ys = [b for b in xs if b > 1]
assert ys == [2, 3]
assert n == 3
