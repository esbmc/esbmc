# #10075: r holds an int from one call and None from another, so it is an
# Optional[int] and `r == 42` must not fold to False on the int path.
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
