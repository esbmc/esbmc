def show(x: float) -> str:
    return "v %f" % x


assert show(2.0000005) == "v 2.000000"
