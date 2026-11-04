def f(y: float) -> str:
    return "%.2f|%.f" % (y, y)


x: float = 2.5
s = "v %.1f" % x
assert s == "v 2.5"
assert "v %.1f" % (x, ) == "v 2.5"
assert "%.0f%%" % x == "2%"
assert f(0.125) == "0.12|0"
assert f(1.75) == "1.75|2"
