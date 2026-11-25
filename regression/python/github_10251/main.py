# Issue #10251: the type is traced to an int, so isinstance() is decided.
def g():
    return 1


def func(a):
    return a()


def run():
    func(lambda: 1)


b = func(g)
run()
assert isinstance(b, int)
