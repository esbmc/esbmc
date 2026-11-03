calls = 0


def g() -> int:
    global calls
    calls += 1
    return calls


x, *ys = g(), g(), g()
assert calls == 3
assert x == 1
assert ys == [2, 3]
