n = 5


def reader() -> bool:
    return isinstance(n, int)


def rebind() -> None:
    global n
    n = [10]


rebind()
assert reader()
