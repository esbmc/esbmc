def f(y: float) -> str:
    return "%.2f|%.f" % (y, y)


x: float = 2.5
assert "v %.1f" % x == "v 2.5"
assert f(1.75) == "1.75|1"
