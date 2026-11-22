# #10109: r holds a str from one call and None from another, so it is an
# Optional[str] and `r == "a"` must hold on the str path.
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
        assert r is None


run(True)
run(False)
