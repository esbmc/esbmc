def show(x: float) -> str:
    return "v %.1f" % x


assert show(0.25) == "v 0.3"
