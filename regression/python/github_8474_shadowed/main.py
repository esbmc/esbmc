# A parameter named like a module function shadows it, so `get()()` must not
# call through the module function's result (#8474).
def f():
    return 1


def g():
    return 2


def get():
    return f


def ret_g():
    return g


def main(get) -> None:
    assert get()() == 2


main(ret_g)
