class P:
    def __init__(self, w: float):
        self.w = w


def half() -> float:
    return 0.5


def show(x: float) -> str:
    return "v %f" % x


def fixed(x: float) -> str:
    return f"{x:.2f}|{x:.0f}"


x = 2.5
if nondet_bool():
    x = 1.5
s = "v %f" % x
assert s == "v 2.500000" or s == "v 1.500000"
assert "%f%%" % half() == "0.500000%"
assert "w=%f" % P(-3.25).w == "w=-3.250000"
assert show(2.0000005) == "v 2.000001"
assert show(1.0 / 3.0) == "v 0.333333"
assert fixed(0.125) == "0.12|0"
assert fixed(2.5) == "2.50|2"
assert fixed(-0.001) == "-0.00|-0"
