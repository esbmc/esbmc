calls: int = 0


def first(t) -> int:
    global calls
    calls += 1
    return t[0] + 1


def name(t) -> str:
    global calls
    calls += 1
    return "a"


skip: bool = False
assert not (skip and isinstance(first((0, 1)), int))
assert calls == 0
assert isinstance(first((0, 1)), int)
assert not isinstance(first((0, 1)), (str, float))
assert isinstance(name((0, 1)), str)
assert calls == 3
