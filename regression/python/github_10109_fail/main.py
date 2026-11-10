# #10109: r holds a str from one call and None from another; on the None
# path it is None.
def f():
    return "a"


def g():
    pass


def run(flag):
    if flag:
        r = f()
    else:
        r = g()
    if flag:
        assert r == "a"
    else:
        assert r is not None


run(True)
run(False)
