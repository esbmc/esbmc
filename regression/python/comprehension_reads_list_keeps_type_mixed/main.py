n = 0


def g() -> int:
    global n
    n += 1
    return n


def f(a):
    return a + 1


xs = [g(), 3]
c = [f(b) for b in xs]
assert c == [2, 4]
assert n == 1
