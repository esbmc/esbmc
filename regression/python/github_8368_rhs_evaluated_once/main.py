n = 0


def pairs() -> list[tuple[int, int]]:
    global n
    n += 1
    return [(1, 2), (3, 4)]


def one() -> int:
    global n
    n += 1
    return 1


(a, b), (c, d) = pairs()
assert n == 1 and a == 1 and d == 4
x, *y = [one(), 2, 3]
assert n == 2 and x == 1 and y[1] == 3


def pair() -> tuple[int, int]:
    global n
    n += 1
    return (5, 6)


(p, q), *r = [pair(), 7]
assert n == 3 and p == 5 and q == 6 and r[0] == 7
