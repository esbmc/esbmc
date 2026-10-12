calls = 0


def make() -> tuple[int, int, int]:
    global calls
    calls += 1
    return (1, 2, 3)


x, *ys = make()
assert calls == 1
assert x == 1
assert ys == [2, 3]
