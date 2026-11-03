import os


def f(a: str, b: str) -> str:
    return os.path.join(a, b)


assert f("x", "/y") == "x/y"
