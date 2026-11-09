def show(x: float) -> str:
    return "v %.1f" % x


def pair(a: float, b: float) -> str:
    return "%.2f/%.f%%" % (a, b)


x: float = 2.5
s = "v %.1f" % x
assert s == "v 2.5"
assert "v %.1f" % (x, ) == s
assert show(1.0 / 3.0) == "v 0.3"
assert show(-2.25) == "v -2.2"
assert pair(0.125, 2.5) == "0.12/2%"
