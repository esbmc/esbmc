import os


def f() -> int:
    os = "a b c"
    words = os.split()
    return len(words)


assert f() == 2
