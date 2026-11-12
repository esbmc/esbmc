# Issue #10251: a tuple of types, with the operand's type traced to an int.
def g():
    return 1


def func(a):
    return a()


def run():
    func(lambda: 1)


b = func(g)
run()
assert isinstance(b, (str, int))
