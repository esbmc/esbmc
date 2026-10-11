count: int = 0


def left() -> int:
    global count
    count += 1
    return -7


r = left() % 5
assert r == 3
assert count >= 2
