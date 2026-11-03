# The path is cut on the assert's evaluated condition, so f runs once.
count = 0


def f() -> int:
    global count
    count += 1
    return 1


assert f() > 0
assert count == 1
