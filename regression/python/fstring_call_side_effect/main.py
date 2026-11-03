n = 0


def g() -> str:
    global n
    n += 1
    return "a"


def h() -> int:
    global n
    n += 10
    return 42


s = f"x{g()}"
assert n == 1
assert s == "xa"

t: str = f"{h():>4}"
assert n == 11

u = f"{g()!r}"
assert n == 12
assert u == "'a'"
