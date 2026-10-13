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


class C:

    def m(self) -> str:
        global n
        n += 1
        return "a"


assert h() == "a"
assert n == 1
assert h() != "b"
assert n == 2
assert "a" == h()
assert n == 3

order = 0
assert h() != g()
assert n == 5
assert order == 12

assert C().m() == "a"
assert n == 6

if h() == "a":
    n += 10
assert n == 17

r = h() == "a"
assert r
assert n == 18

x = 1 if h() == "a" else 2
assert x == 1
assert n == 19

b = False
assert not (b and h() == "a")
assert n == 19
assert b or h() == "a"
assert n == 20

assert h() < "b"
assert n == 21

f = lambda: h() == "a"
assert f()
assert n == 22
