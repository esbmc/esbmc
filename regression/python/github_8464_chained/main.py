n = 0
order = 0


def h() -> str:
    global n, order
    n += 1
    order = order * 10 + 1
    return "a"


def g() -> str:
    global n, order
    n += 1
    order = order * 10 + 2
    return "b"


assert "a" == h() == "a"
assert n == 1
assert "0" < h() < "b"
assert n == 2

order = 0
assert "a" == h() == "a" != g()
assert n == 4
assert order == 12

order = 0
assert not ("b" == h() == g())
assert n == 5
assert order == 1

i = 0


def w() -> str:
    global i
    i += 1
    return "x" if i < 4 else "y"


while "x" == w() == "x":
    pass
assert i == 4
