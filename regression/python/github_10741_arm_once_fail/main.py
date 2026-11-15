# k() runs once, not twice.
count = 0


def k():
    global count
    count += 1
    return 3


def g():
    return 7


flag = nondet_bool()
h = g if flag else k()
if flag:
    assert h() == 7
    assert count == 0
else:
    assert count == 2
