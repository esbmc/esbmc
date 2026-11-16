def f(c: bool):
    n: int = chr(97)
    xs = [n]
    if c:
        return xs[0]
    return 5


assert f(False) == 5
