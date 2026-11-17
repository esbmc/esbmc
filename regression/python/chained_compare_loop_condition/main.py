calls = 0


def f(x: int) -> int:
    global calls
    calls += 1
    return 1


i = 0
while 0 < i + f(i) < 3:
    i += 1
assert i == 2
assert calls == 3
