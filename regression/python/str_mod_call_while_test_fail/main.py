n = 0


def f() -> int:
    global n
    n += 1
    return n


while "%d" % f() != "3":
    pass
assert n == 2
