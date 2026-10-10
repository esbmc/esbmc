n = 0


def g() -> str:
    global n
    n += 1
    return "a"


s = f"x{g()}"
assert n == 0
