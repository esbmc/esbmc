def g(b: list[str], c: list[str]) -> str:
    return ".".join(b + c)


def h(b: list[str], c: list[str]) -> str:
    return "".join(b + c)


def k(b: list[str]) -> str:
    return "-".join(b + ["z"])


assert g(["a"], ["b"]) == "a.b"
assert h(["x"], ["y"]) == "xy"
assert k(["y"]) == "y-z"
