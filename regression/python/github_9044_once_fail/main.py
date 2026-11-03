# isinstance() evaluates a side-effecting Optional operand once (#9044)
cnt = 0


def g() -> int:
    global cnt
    cnt += 1
    return 3


c = 1
r = isinstance(g() if c == 1 else None, int)
assert cnt != 1
