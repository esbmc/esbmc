calls: int = 0


def one() -> int:
    global calls
    calls += 1
    return 1


x: int = 0
ok: bool = x != 0 and 10 // one() > 1
assert calls == 1
