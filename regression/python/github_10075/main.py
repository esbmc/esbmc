def f():
    return 42


def g():
    pass


def run(flag):
    if flag:
        r = f()
    else:
        r = g()
    if flag:
        assert r == 42
    else:
        assert r is None


run(True)
run(False)
