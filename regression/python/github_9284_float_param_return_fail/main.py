def f(c: bool, x: float):
    if c:
        return "d"
    return x


assert f(True, 2.5) != "d"
