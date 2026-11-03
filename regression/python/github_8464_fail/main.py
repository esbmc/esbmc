n = 0


def h() -> str:
    global n
    n += 1
    return "a"


assert h() == "a"
assert n == 2
