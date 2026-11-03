def f(a: str) -> str:
    return "-".join([a, "b"])


assert f("xy") == "xy-b"
