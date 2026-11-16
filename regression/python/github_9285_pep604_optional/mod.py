def func(x: int | None = None):
    if x is None:
        return "a"
    return x


def gunc(x: None | int = None):
    if x is None:
        return "a"
    return x


def hunc(x: int | str):
    return x
