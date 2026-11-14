def f():
    n: int = chr(97)
    return n


assert f() != "a"
