count: int = 0


def base() -> int:
    global count
    count += 1
    return 3


r = base() ** 3
assert r == 27
assert count == 3
