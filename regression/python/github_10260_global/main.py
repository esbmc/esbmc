# Issue #10260: a call that rebinds the global leaves the loop on the original list.
def g(k: int) -> list[int]:
    r: list[int] = []
    i = 0
    while i < k:
        r.append(9)
        i = i + 1
    return r


xs: list[int] = g(3)


def h() -> None:
    global xs
    xs = g(5)


n = 0
for x in xs:
    h()
    n = n + 1
assert n == 3
