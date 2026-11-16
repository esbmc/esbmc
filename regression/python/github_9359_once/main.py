# Issue #9359: each operand of an int/float comparison is evaluated once.
count = 0


def f() -> float:
    global count
    count += 1
    return float(count)


assert f() == 1
assert count == 1
assert 2 == f()
assert count == 2
assert f() <= 3
assert count == 3
