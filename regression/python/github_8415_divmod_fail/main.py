count: int = 0


def left() -> int:
    global count
    count += 1
    return -7


q, r = divmod(left(), 5)
assert r == 3
assert count >= 2
