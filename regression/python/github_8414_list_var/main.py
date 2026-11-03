def f(a: str) -> str:
    xs = [a, "b"]
    a = "zz"
    return "-".join(xs)


assert f("xy") == "xy-b"
