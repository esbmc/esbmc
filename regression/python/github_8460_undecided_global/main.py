import string

string = "a b"


def f() -> int:
    return len(string.split())


assert f() == 2
