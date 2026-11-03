# A parameter named like a module function shadows it, so `get()()` must not
# call through the module function's result (#8474). The argument is a lambda,
# so the parameter is not specialised to a module function.
def f():
    return 1


def g():
    return 2


def get():
    return f


def main(get) -> None:
    assert get()() == 2


main(lambda: g)
