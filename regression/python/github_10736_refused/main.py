def f():
    n: int = chr(97)
    xs = [n]
    return xs[0]


assert f() == "a"
