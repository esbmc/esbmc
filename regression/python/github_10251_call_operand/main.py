# Issue #10251: the operand is a call whose int type is traced after it runs.
def g():
    return 1


def func(a):
    return a()


def run():
    func(lambda: 1)


b = func(g)
run()
assert isinstance(func(g), int)
