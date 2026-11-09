def f():
    return 42


def g() -> None:
    return 2.5


def run(flag):
    if flag:
        r = f()
    else:
        r = g()
    assert r == 42 or r == 2.5


run(True)
run(False)
